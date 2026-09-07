use crate::buffers::StageBuffers;
use crate::shape_pool::{PoolKind, ShapePool, height_field_triangles};
use crate::stages::{Stages, build_stages, encode_physics};
use bytemuck::Zeroable;
use dynamis_gpu::GpuContext;
use dynamis_layout::{
    BodyCommandRecord, BODY_CCD, BODY_SLEEPING, ColliderRecord, ConstraintCommandRecord,
    ConstraintRecord, DispatchArgs, PATCH_ANGULAR_VELOCITY, PATCH_CCD, PATCH_COLLIDER,
    PATCH_FRICTION, PATCH_GROUP, PATCH_INVERSE_MASS, PATCH_KINEMATIC, PATCH_MASK,
    PATCH_ORIENTATION, PATCH_POSITION, PATCH_RESTITUTION, PATCH_VELOCITY, QueryRecord,
    QueryResultHeader, RigidBodyRecord, SimParamsRecord,
};
use dynamis_model::{
    BodyDesc, BodyHandle, BodyState, ColliderDesc, ConstraintDesc, ConstraintHandle,
    ContactEvent, ContactEventKind, PhysicsConfig, QueryFilter, Shape, ShapeSourceHandle,
    inverse_inertia_diagonal,
};
use dynamis_query::{QueryHandle, QueryHit, QueryPool};
use std::collections::VecDeque;

const PAIR_CAPACITY_PER_BODY: usize = 64;
const QUERY_RATIO: usize = 2;
const MAX_COLLIDERS_PER_BODY_USIZE: usize = 4;

pub struct Simulation {
    gpu: GpuContext,
    config: PhysicsConfig,
    capacity: usize,
    constraint_capacity: usize,
    step_index: u64,
    alive: Vec<BodyHandle>,
    index_of: Vec<u32>,
    generations: Vec<u32>,
    free_ids: Vec<u32>,
    commands: Vec<BodyCommandRecord>,
    collider_descs: Vec<Vec<ColliderDesc>>,
    masses: Vec<f32>,
    constraint_alive: Vec<ConstraintHandle>,
    constraint_index_of: Vec<u32>,
    constraint_generations: Vec<u32>,
    constraint_free_ids: Vec<u32>,
    constraint_commands: Vec<ConstraintCommandRecord>,
    queries: Vec<QueryRecord>,
    query_pool: QueryPool,
    shape_pool: ShapePool,
    events: Vec<ContactEvent>,
    pending_events: VecDeque<(u64, Vec<u8>)>,
    event_capacity: usize,
    buffers: StageBuffers,
    stages: Stages,
    states: Vec<Option<BodyState>>,
    constraint_records: Vec<ConstraintRecord>,
}

impl Simulation {
    pub fn new(gpu: GpuContext, capacity: usize, config: PhysicsConfig) -> Self {
        Self::with_shape_sources(gpu, capacity, capacity, config)
    }

    pub fn with_shape_sources(
        gpu: GpuContext,
        capacity: usize,
        shape_sources: usize,
        config: PhysicsConfig,
    ) -> Self {
        assert!(capacity > 0, "simulation capacity must be positive");
        let pair_capacity = capacity * PAIR_CAPACITY_PER_BODY;
        let query_capacity = capacity * QUERY_RATIO;
        let constraint_capacity = capacity;
        let buffers = StageBuffers::new(
            gpu.device(),
            shape_sources,
            capacity,
            pair_capacity,
            query_capacity,
            constraint_capacity,
        );
        buffers
            .prev_contact_count
            .write(gpu.queue(), bytemuck::cast_slice(&[DispatchArgs::none()]));
        let stages = build_stages(gpu.device(), &buffers, capacity as u32);
        let query_header_bytes = query_capacity * std::mem::size_of::<QueryResultHeader>();
        let zeros = vec![0u8; query_header_bytes];
        buffers.query_headers.write(gpu.queue(), &zeros);
        Self {
            gpu,
            config,
            capacity,
            constraint_capacity,
            step_index: 0,
            alive: Vec::new(),
            index_of: vec![u32::MAX; capacity],
            generations: vec![1; capacity],
            free_ids: (0..capacity as u32).rev().collect(),
            commands: Vec::new(),
            collider_descs: Vec::new(),
            masses: Vec::new(),
            constraint_alive: Vec::new(),
            constraint_index_of: vec![u32::MAX; capacity],
            constraint_generations: vec![1; capacity],
            constraint_free_ids: (0..capacity as u32).rev().collect(),
            constraint_commands: Vec::new(),
            queries: Vec::new(),
            query_pool: QueryPool::new(query_capacity),
            shape_pool: ShapePool::new(shape_sources),
            events: Vec::new(),
            pending_events: VecDeque::new(),
            event_capacity: pair_capacity,
            buffers,
            stages,
            states: Vec::new(),
            constraint_records: Vec::new(),
        }
    }

    pub fn config(&self) -> &PhysicsConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: PhysicsConfig) {
        self.config = config;
    }

    pub fn set_gravity(&mut self, gravity: [f32; 3]) {
        self.config.gravity = gravity;
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn bodies(&self) -> &[BodyHandle] {
        &self.alive
    }

    pub fn count(&self) -> usize {
        self.alive.len()
    }

    pub fn grow(&mut self, capacity: usize) {
        assert!(
            capacity >= self.alive.len(),
            "grow capacity must not drop below the live body count"
        );
        if capacity == self.capacity {
            return;
        }
        let pair_capacity = capacity * PAIR_CAPACITY_PER_BODY;
        let query_capacity = capacity * QUERY_RATIO;
        let buffers = StageBuffers::new(
            self.gpu.device(),
            self.shape_pool.capacity(),
            capacity,
            pair_capacity,
            query_capacity,
            self.constraint_capacity,
        );
        buffers
            .prev_contact_count
            .write(self.gpu.queue(), bytemuck::cast_slice(&[DispatchArgs::none()]));
        self.shape_pool.reset_upload_cursor();
        self.shape_pool.upload_pending(
            self.gpu.queue(),
            &buffers.shapes,
            &buffers.shape_vertices,
            &buffers.shape_triangles,
            &buffers.shape_nodes,
        );
        self.sync_state_to(&buffers, capacity);
        let stages = build_stages(self.gpu.device(), &buffers, capacity as u32);
        self.buffers = buffers;
        self.stages = stages;
        self.capacity = capacity;
    }

    fn sync_state_to(&mut self, buffers: &StageBuffers, capacity: usize) {
        let mut bodies = Vec::with_capacity(self.alive.len());
        let mut colliders = Vec::new();
        for (slot, handle) in self.alive.iter().enumerate() {
            let id = handle.id as usize;
            let desc = self.build_desc(id);
            let inertia = self.compound_solid_inertia(&desc);
            let mut body_record = RigidBodyRecord::build(
                &desc,
                handle.id,
                handle.generation,
                inertia,
            );
            if let Some(state) = self.state_snapshot(id) {
                body_record.position = state.position;
                body_record.prev_position = state.position;
                body_record.orientation = state.orientation;
                body_record.velocity = state.velocity;
                body_record.angular_velocity = state.angular_velocity;
                body_record.flags |= if state.sleeping { BODY_SLEEPING } else { 0 };
            }
            bodies.push(body_record);
            for index in 0..MAX_COLLIDERS_PER_BODY_USIZE {
                let collider = self
                    .collider_descs
                    .get(id)
                    .and_then(|descs| descs.get(index))
                    .map(|desc| self.collider_record(desc))
                    .unwrap_or_else(ColliderRecord::zeroed);
                colliders.push(collider);
            }
            let _ = slot;
        }
        buffers
            .bodies_current
            .write(self.gpu.queue(), bytemuck::cast_slice(&bodies));
        buffers
            .colliders
            .write(self.gpu.queue(), bytemuck::cast_slice(&colliders));
        let constraint_count = self.constraint_alive.len() as u32;
        let mut constraints = Vec::with_capacity(self.constraint_alive.len());
        for handle in &self.constraint_alive {
            let index = self.constraint_index_of[handle.id as usize] as usize;
            constraints.push(self.constraint_records[index]);
        }
        if !constraints.is_empty() {
            buffers
                .constraints
                .write(self.gpu.queue(), bytemuck::cast_slice(&constraints));
        }
        buffers.constraint_count_state.write(
            self.gpu.queue(),
            bytemuck::cast_slice(&[DispatchArgs::sized(constraint_count)]),
        );
        let _ = capacity;
    }

    fn build_desc(&self, id: usize) -> BodyDesc {
        let descs = &self.collider_descs[id];
        BodyDesc {
            colliders: descs.clone(),
            position: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            mass: self.masses[id],
            collision_group: 0,
            collision_mask: 0,
            kinematic: false,
            ccd: false,
        }
    }

    fn state_snapshot(&self, id: usize) -> Option<BodyState> {
        self.states.get(id).cloned().flatten()
    }

    fn collider_record(&self, desc: &ColliderDesc) -> ColliderRecord {
        let source = match desc.shape {
            Shape::Hull(handle) | Shape::Mesh(handle) | Shape::HeightField(handle) => handle.id,
            _ => 0,
        };
        ColliderRecord::build(desc, source)
    }

    fn compound_solid_inertia(&self, desc: &BodyDesc) -> [f32; 3] {
        let inverse_mass = layout_inverse_mass(desc);
        if inverse_mass == 0.0 {
            return [0.0; 3];
        }
        let solid = desc
            .colliders
            .iter()
            .filter(|collider| !collider.sensor)
            .collect::<Vec<_>>();
        let count = solid.len().max(1) as f32;
        let mass = 1.0 / inverse_mass;
        let mut sum = [0.0f32; 3];
        for collider in solid {
            let fraction = inverse_inertia_diagonal(
                &collider.shape,
                1.0 / (mass / count),
                self.shape_bounds(&collider.shape),
            );
            sum[0] += 1.0 / fraction[0];
            sum[1] += 1.0 / fraction[1];
            sum[2] += 1.0 / fraction[2];
        }
        [1.0 / sum[0], 1.0 / sum[1], 1.0 / sum[2]]
    }

    fn shape_bounds(&self, shape: &Shape) -> Option<([f32; 3], [f32; 3])> {
        match shape {
            Shape::Hull(handle) | Shape::Mesh(handle) | Shape::HeightField(handle) => {
                Some(self.shape_pool.record(*handle).bounds)
            }
            _ => None,
        }
    }

    pub fn add_hull(&mut self, vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> ShapeSourceHandle {
        self.allocate_shape(PoolKind::Hull, vertices, triangles.to_vec())
    }

    pub fn add_mesh(&mut self, vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> ShapeSourceHandle {
        self.allocate_shape(PoolKind::Mesh, vertices, triangles.to_vec())
    }

    pub fn add_height_field(
        &mut self,
        rows: u32,
        cols: u32,
        heights: &[f32],
        cell_size: [f32; 2],
    ) -> ShapeSourceHandle {
        let (vertices, triangles) = height_field_triangles(rows, cols, heights, cell_size);
        self.allocate_shape(PoolKind::HeightField, &vertices, triangles)
    }

    fn allocate_shape(
        &mut self,
        kind: PoolKind,
        vertices: &[[f32; 3]],
        triangles: Vec<[u32; 3]>,
    ) -> ShapeSourceHandle {
        let handle = self.shape_pool.allocate(kind, vertices, &triangles);
        self.shape_pool.upload_pending(
            self.gpu.queue(),
            &self.buffers.shapes,
            &self.buffers.shape_vertices,
            &self.buffers.shape_triangles,
            &self.buffers.shape_nodes,
        );
        handle
    }

    pub fn bodies_buffer(&self) -> &wgpu::Buffer {
        self.buffers.bodies_current.buffer()
    }

    pub fn colliders_buffer(&self) -> &wgpu::Buffer {
        self.buffers.colliders.buffer()
    }

    pub fn commands_buffer(&self) -> &wgpu::Buffer {
        self.buffers.commands.buffer()
    }

    pub fn debug_constraint_keys_a(&self) -> &wgpu::Buffer {
        self.buffers.constraint_gather_a_keys_out.buffer()
    }

    pub fn debug_constraint_values_a(&self) -> &wgpu::Buffer {
        self.buffers.constraint_gather_a_values_out.buffer()
    }

    pub fn debug_constraint_first_b(&self) -> &wgpu::Buffer {
        self.buffers.constraint_first_b.buffer()
    }

    pub fn debug_constraint_keys_b_out(&self) -> &wgpu::Buffer {
        self.buffers.constraint_gather_b_keys_out.buffer()
    }

    pub fn debug_constraint_values_b(&self) -> &wgpu::Buffer {
        self.buffers.constraint_gather_b_values_out.buffer()
    }

    pub fn debug_constraint_first_a(&self) -> &wgpu::Buffer {
        self.buffers.constraint_first_a.buffer()
    }

    pub fn debug_constraint_deltas(&self) -> &wgpu::Buffer {
        self.buffers.constraint_deltas.buffer()
    }

    pub fn debug_constraint_command_count(&self) -> &wgpu::Buffer {
        self.buffers.constraint_command_count.buffer()
    }

    pub fn debug_constraints(&self) -> &wgpu::Buffer {
        self.buffers.constraints.buffer()
    }

    pub fn debug_contact_count(&self) -> &wgpu::Buffer {
        self.buffers.contact_count.buffer()
    }

    pub fn debug_pair_count(&self) -> &wgpu::Buffer {
        self.buffers.pair_count.buffer()
    }

    pub fn debug_entry_keys(&self) -> &wgpu::Buffer {
        self.buffers.entries.keys_hi.buffer()
    }

    pub fn debug_entry_lo(&self) -> &wgpu::Buffer {
        self.buffers.entries.keys_lo.buffer()
    }

    pub fn debug_contact_valid(&self) -> &wgpu::Buffer {
        self.buffers.contact_valid.buffer()
    }

    pub fn debug_pair_args(&self) -> &wgpu::Buffer {
        self.buffers.pair_args.buffer()
    }

    pub fn debug_compact_ranks(&self) -> &wgpu::Buffer {
        self.buffers.compact_ranks.buffer()
    }

    pub fn debug_compact_offsets(&self) -> &wgpu::Buffer {
        self.buffers.compact_block_offsets.buffer()
    }

    pub fn debug_a_bodies(&self) -> &wgpu::Buffer {
        self.buffers.contact_a_body.buffer()
    }

    pub fn debug_compact_sums(&self) -> &wgpu::Buffer {
        self.buffers.compact_block_sums.buffer()
    }

    pub fn debug_compact_sums_pad(&self) -> &wgpu::Buffer {
        self.buffers.compact_block_sums_pad.buffer()
    }

    pub fn debug_contacts(&self) -> &wgpu::Buffer {
        self.buffers.contacts.buffer()
    }

    pub fn debug_prev_contacts(&self) -> &wgpu::Buffer {
        self.buffers.prev_contacts.buffer()
    }

    pub fn debug_events(&self) -> &wgpu::Buffer {
        self.buffers.events.buffer()
    }

    pub fn debug_event_count(&self) -> &wgpu::Buffer {
        self.buffers.event_count.buffer()
    }

    pub fn debug_pair_lo(&self) -> &wgpu::Buffer {
        self.buffers.pairs.keys_lo.buffer()
    }

    pub fn debug_pair_hi(&self) -> &wgpu::Buffer {
        self.buffers.pairs.keys_hi.buffer()
    }

    pub fn debug_entry_count(&self) -> &wgpu::Buffer {
        self.buffers.entry_count.buffer()
    }

    pub fn debug_large_count(&self) -> &wgpu::Buffer {
        self.buffers.large_count.buffer()
    }

    pub fn debug_large_bodies(&self) -> &wgpu::Buffer {
        self.buffers.large_bodies.buffer()
    }

    pub fn debug_aabbs(&self) -> &wgpu::Buffer {
        self.buffers.aabbs.buffer()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.gpu.queue()
    }

    pub fn device(&self) -> &wgpu::Device {
        self.gpu.device()
    }

    pub fn spawn(&mut self, desc: BodyDesc) -> BodyHandle {
        let id = self
            .free_ids
            .pop()
            .expect("simulation body capacity exhausted");
        self.generations[id as usize] += 1;
        let handle = BodyHandle {
            id,
            generation: self.generations[id as usize],
        };
        let slot = self.alive.len() as u32;
        self.index_of[id as usize] = slot;
        self.alive.push(handle);
        self.validate_world_geometry(&desc);
        self.masses.resize(self.alive.len(), 1.0);
        self.masses[id as usize] = desc.mass;
        self.collider_descs.resize(self.alive.len(), Vec::new());
        self.collider_descs[id as usize] = desc.colliders.clone();
        let inertia = self.compound_solid_inertia(&desc);
        let body = RigidBodyRecord::build(&desc, handle.id, handle.generation, inertia);
        let colliders = self.collider_block(&desc);
        self.record_state(id as usize, &desc, &body);
        self.commands
            .push(BodyCommandRecord::add(slot, body, colliders));
        handle
    }

    fn collider_block(&self, desc: &BodyDesc) -> [ColliderRecord; 4] {
        let mut colliders = [ColliderRecord::zeroed(); 4];
        for (index, collider) in desc.colliders.iter().enumerate() {
            colliders[index] = self.collider_record(collider);
        }
        colliders
    }

    fn record_state(&mut self, id: usize, desc: &BodyDesc, body: &RigidBodyRecord) {
        if self.states.len() <= id {
            self.states.resize(id + 1, None);
        }
        self.states[id] = Some(BodyState {
            position: desc.position,
            orientation: desc.orientation,
            velocity: desc.velocity,
            angular_velocity: desc.angular_velocity,
            inverse_mass: body.inverse_mass,
            sleeping: false,
            step: self.step_index,
        });
    }

    fn validate_world_geometry(&self, desc: &BodyDesc) {
        for collider in &desc.colliders {
            if collider.shape.is_world_geometry() && (desc.mass > 0.0 && !desc.kinematic) {
                panic!("world geometry colliders must be static or kinematic");
            }
        }
    }

    pub fn remove(&mut self, handle: BodyHandle) {
        self.validate(handle);
        self.assert_no_constraints(handle);
        let id = handle.id as usize;
        let slot = self.index_of[id] as usize;
        let tail = self.alive.len() - 1;
        let moved = self.alive[tail];
        self.alive.swap_remove(slot);
        self.index_of[moved.id as usize] = slot as u32;
        self.index_of[id] = u32::MAX;
        self.free_ids.push(handle.id);
        self.collider_descs[id] = Vec::new();
        self.states[id] = None;
        self.commands
            .push(BodyCommandRecord::remove(slot as u32, tail as u32));
    }

    pub fn set_position(&mut self, handle: BodyHandle, position: [f32; 3]) {
        self.validate(handle);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.position = position;
        }
        let mut record = RigidBodyRecord::zeroed();
        record.position = position;
        self.schedule_patch(handle, PATCH_POSITION, record);
    }

    pub fn set_orientation(&mut self, handle: BodyHandle, orientation: [f32; 4]) {
        self.assert_unit(orientation);
        self.validate(handle);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.orientation = orientation;
        }
        let mut record = RigidBodyRecord::zeroed();
        record.orientation = orientation;
        self.schedule_patch(handle, PATCH_ORIENTATION, record);
    }

    pub fn set_velocity(&mut self, handle: BodyHandle, velocity: [f32; 3]) {
        self.validate(handle);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.velocity = velocity;
        }
        let mut record = RigidBodyRecord::zeroed();
        record.velocity = velocity;
        self.schedule_patch(handle, PATCH_VELOCITY, record);
    }

    pub fn set_angular_velocity(&mut self, handle: BodyHandle, angular_velocity: [f32; 3]) {
        self.validate(handle);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.angular_velocity = angular_velocity;
        }
        let mut record = RigidBodyRecord::zeroed();
        record.angular_velocity = angular_velocity;
        self.schedule_patch(handle, PATCH_ANGULAR_VELOCITY, record);
    }

    pub fn set_mass(&mut self, handle: BodyHandle, mass: f32) {
        assert!(mass >= 0.0, "mass must be non-negative");
        self.validate(handle);
        self.masses[handle.id as usize] = mass;
        let asc = self.description(handle);
        let inverse_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        let mut record = RigidBodyRecord::zeroed();
        record.inverse_mass = inverse_mass;
        record.inverse_inertia_body = self.compound_solid_inertia(&asc);
        if let Some(state) = self.states[handle.id as usize].as_mut() {
            state.inverse_mass = inverse_mass;
        }
        self.schedule_patch(handle, PATCH_INVERSE_MASS, record);
    }

    pub fn set_collider(&mut self, handle: BodyHandle, index: usize, collider: ColliderDesc) {
        self.validate(handle);
        assert!(
            index < MAX_COLLIDERS_PER_BODY_USIZE,
            "collider index out of range"
        );
        self.collider_descs[handle.id as usize][index] = collider;
        let asc = self.description(handle);
        let mut record = RigidBodyRecord::zeroed();
        record.inverse_mass = self
            .states[handle.id as usize]
            .map(|state| state.inverse_mass)
            .unwrap_or(0.0);
        record.inverse_inertia_body = self.compound_solid_inertia(&asc);
        let record_collider = self.collider_record(&collider);
        self.schedule_patch_collider(handle, PATCH_COLLIDER, record, record_collider, index as u32);
    }

    pub fn set_shape(&mut self, handle: BodyHandle, shape: Shape) {
        self.validate(handle);
        let collider = &self.collider_descs[handle.id as usize];
        let existing = collider[0];
        self.set_collider(
            handle,
            0,
            ColliderDesc {
                shape,
                offset: existing.offset,
                rotation: existing.rotation,
                friction: existing.friction,
                restitution: existing.restitution,
                sensor: existing.sensor,
            },
        );
    }

    pub fn set_restitution(&mut self, handle: BodyHandle, restitution: f32) {
        self.validate(handle);
        let mut record = RigidBodyRecord::zeroed();
        record.restitution = restitution;
        self.schedule_patch(handle, PATCH_RESTITUTION, record);
    }

    pub fn set_friction(&mut self, handle: BodyHandle, friction: f32) {
        assert!(friction >= 0.0, "friction must be non-negative");
        self.validate(handle);
        let mut record = RigidBodyRecord::zeroed();
        record.friction = friction;
        self.schedule_patch(handle, PATCH_FRICTION, record);
    }

    pub fn set_collision_group(&mut self, handle: BodyHandle, group: u32) {
        self.validate(handle);
        let mut record = RigidBodyRecord::zeroed();
        record.collision_group = group;
        self.schedule_patch(handle, PATCH_GROUP, record);
    }

    pub fn set_collision_mask(&mut self, handle: BodyHandle, mask: u32) {
        self.validate(handle);
        let mut record = RigidBodyRecord::zeroed();
        record.collision_mask = mask;
        self.schedule_patch(handle, PATCH_MASK, record);
    }

    pub fn set_kinematic(&mut self, handle: BodyHandle, kinematic: bool) {
        self.validate(handle);
        let asc = self.description(handle);
        let mut record = RigidBodyRecord::zeroed();
        record.flags = if kinematic { dynamis_layout::BODY_KINEMATIC } else { 0 };
        record.inverse_mass = if kinematic { 0.0 } else { self.inverse_mass_of(&asc) };
        record.inverse_inertia_body = self.compound_solid_inertia(&asc);
        self.schedule_patch(handle, PATCH_KINEMATIC, record);
    }

    pub fn set_ccd(&mut self, handle: BodyHandle, ccd: bool) {
        self.validate(handle);
        let mut record = RigidBodyRecord::zeroed();
        record.flags = if ccd { BODY_CCD } else { 0 };
        self.schedule_patch(handle, PATCH_CCD, record);
    }

    pub fn apply_force(&mut self, handle: BodyHandle, force: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::force(slot, force));
    }

    pub fn apply_force_at_point(
        &mut self,
        handle: BodyHandle,
        force: [f32; 3],
        point: [f32; 3],
    ) {
        let slot = self.command_slot(handle);
        self.commands
            .push(BodyCommandRecord::force_at_point(slot, force, point));
    }

    pub fn apply_torque(&mut self, handle: BodyHandle, torque: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::torque(slot, torque));
    }

    pub fn apply_impulse(&mut self, handle: BodyHandle, impulse: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.commands
            .push(BodyCommandRecord::impulse(slot, impulse));
    }

    pub fn apply_impulse_at_point(
        &mut self,
        handle: BodyHandle,
        impulse: [f32; 3],
        point: [f32; 3],
    ) {
        let slot = self.command_slot(handle);
        self.commands
            .push(BodyCommandRecord::impulse_at_point(slot, impulse, point));
    }

    pub fn apply_angular_impulse(&mut self, handle: BodyHandle, impulse: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.commands
            .push(BodyCommandRecord::angular_impulse(slot, impulse));
    }

    pub fn add_constraint(
        &mut self,
        first: BodyHandle,
        second: BodyHandle,
        desc: ConstraintDesc,
    ) -> ConstraintHandle {
        self.validate(first);
        self.validate(second);
        if first == second {
            panic!("constraint bodies must be distinct");
        }
        self.validate_constraint_desc(&desc);
        let id = self
            .constraint_free_ids
            .pop()
            .expect("simulation constraint capacity exhausted");
        self.constraint_generations[id as usize] += 1;
        let handle = ConstraintHandle {
            id,
            generation: self.constraint_generations[id as usize],
        };
        let slot = self.constraint_alive.len() as u32;
        self.constraint_index_of[id as usize] = slot;
        self.constraint_alive.push(handle);
        let record = ConstraintRecord::build(
            &desc,
            self.index_of[first.id as usize],
            self.index_of[second.id as usize],
        );
        self.constraint_records.push(record);
        self.constraint_commands
            .push(ConstraintCommandRecord::add(slot, record));
        handle
    }

    fn validate_constraint_desc(&self, desc: &ConstraintDesc) {
        if !matches!(
            desc.kind,
            dynamis_model::ConstraintKind::Ball | dynamis_model::ConstraintKind::Distance
        ) && desc.axis == [0.0; 3]
        {
            panic!("constraint axis must be non-zero");
        }
    }

    pub fn remove_constraint(&mut self, handle: ConstraintHandle) {
        self.validate_constraint(handle);
        let id = handle.id as usize;
        let slot = self.constraint_index_of[id] as usize;
        let tail = self.constraint_alive.len() - 1;
        let moved = self.constraint_alive[tail];
        self.constraint_alive.swap_remove(slot);
        self.constraint_index_of[moved.id as usize] = slot as u32;
        self.constraint_index_of[id] = u32::MAX;
        self.constraint_free_ids.push(handle.id);
        self.constraint_records.remove(slot);
        self.constraint_commands
            .push(ConstraintCommandRecord::remove(slot as u32));
    }

    pub fn constraints(&self) -> &[ConstraintHandle] {
        &self.constraint_alive
    }

    pub fn raycast(
        &mut self,
        origin: [f32; 3],
        direction: [f32; 3],
        max_t: f32,
        filter: &QueryFilter,
    ) -> QueryHandle {
        assert!(max_t > 0.0, "raycast distance must be positive");
        assert!(direction != [0.0; 3], "raycast direction must be non-zero");
        self.submit_query(QueryRecord::ray(origin, direction, max_t, filter))
    }

    pub fn sphere_query(
        &mut self,
        center: [f32; 3],
        radius: f32,
        filter: &QueryFilter,
    ) -> QueryHandle {
        assert!(radius > 0.0, "sphere query radius must be positive");
        self.submit_query(QueryRecord::sphere(center, radius, filter))
    }

    pub fn box_query(
        &mut self,
        center: [f32; 3],
        half_extents: [f32; 3],
        filter: &QueryFilter,
    ) -> QueryHandle {
        assert!(
            half_extents.iter().all(|extent| *extent > 0.0),
            "box query half extents must be strictly positive"
        );
        self.submit_query(QueryRecord::box_query(center, half_extents, filter))
    }

    pub fn sweep_query(
        &mut self,
        shape: &Shape,
        orientation: [f32; 4],
        start: [f32; 3],
        direction: [f32; 3],
        length: f32,
        filter: &QueryFilter,
    ) -> QueryHandle {
        assert!(length > 0.0, "sweep length must be positive");
        assert!(direction != [0.0; 3], "sweep direction must be non-zero");
        self.assert_unit(orientation);
        assert!(
            shape.is_convex(),
            "sweep queries require a convex shape"
        );
        self.submit_query(QueryRecord::sweep(
            shape,
            orientation,
            start,
            direction,
            length,
            filter,
        ))
    }

    fn submit_query(&mut self, record: QueryRecord) -> QueryHandle {
        let slot = self.query_pool.allocate();
        let generation = self.query_pool.generation(slot);
        let mut record = record;
        record.slot = slot as u32;
        record.max_hits = record.max_hits.min(dynamis_layout::MAX_HITS_PER_QUERY);
        self.queries.push(record);
        QueryHandle {
            slot: slot as u32,
            generation,
        }
    }

    pub fn query_hit(&self, handle: QueryHandle) -> Option<QueryHit> {
        self.validate_query(handle);
        self.query_pool.hit(handle.slot as usize)
    }

    pub fn query_hits(&self, handle: QueryHandle) -> &[QueryHit] {
        self.validate_query(handle);
        self.query_pool.hits(handle.slot as usize)
    }

    pub fn query_overflow(&self, handle: QueryHandle) -> bool {
        self.validate_query(handle);
        self.query_pool.overflow(handle.slot as usize)
    }

    pub fn drain_events(&mut self) -> Vec<ContactEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn step(&mut self, dt: f32) {
        assert!(dt > 0.0, "timestep must be strictly positive");
        let step = self.step_index;
        self.collect_readbacks();
        let queue = self.gpu.queue();
        let device = self.gpu.device();
        self.buffers
            .commands
            .write(queue, bytemuck::cast_slice(&self.commands));
        self.buffers.command_count.write(
            queue,
            bytemuck::cast_slice(&[DispatchArgs::sized(self.commands.len() as u32)]),
        );
        self.buffers
            .constraint_commands
            .write(queue, bytemuck::cast_slice(&self.constraint_commands));
        self.buffers.constraint_command_count.write(
            queue,
            bytemuck::cast_slice(&[DispatchArgs::sized(self.constraint_commands.len() as u32)]),
        );
        self.buffers.constraint_count_state.write(
            queue,
            bytemuck::cast_slice(&[DispatchArgs::sized(self.constraint_alive.len() as u32)]),
        );
        let params = SimParamsRecord::new(
            &self.config,
            dt,
            self.alive.len() as u32,
            self.constraint_alive.len() as u32,
        );
        self.buffers
            .params
            .write(queue, bytemuck::cast_slice(&[params]));
        self.buffers.reset_counters(queue);
        let header_bytes = self.query_pool.capacity() * std::mem::size_of::<QueryResultHeader>();
        self.buffers
            .query_headers
            .write(queue, &vec![0u8; header_bytes]);
        if !self.queries.is_empty() {
            self.buffers
                .queries
                .write(queue, bytemuck::cast_slice(&self.queries));
            let slots = self.queries.iter().map(|query| query.slot).collect();
            self.query_pool.mark_batch(step, slots);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dynamis step encoder"),
        });

        encode_physics(
            &self.stages,
            &self.buffers,
            device,
            &mut encoder,
            self.alive.len() as u32,
            self.config.solve_iterations,
            self.config.position_iterations,
            self.island_rounds(),
            self.queries.len() as u32,
            self.constraint_alive.len() as u32,
        );

        let stale_bodies = self.buffers.bodies_readback.enqueue(
            device,
            &mut encoder,
            self.buffers.bodies_current.buffer(),
            step,
        );
        let stale_queries = if self.queries.is_empty() {
            None
        } else {
            let header_bytes = self.query_pool.capacity() * std::mem::size_of::<QueryResultHeader>();
            encoder.copy_buffer_to_buffer(
                self.buffers.query_headers.buffer(),
                0,
                self.buffers.query_pack.buffer(),
                0,
                header_bytes as u64,
            );
            encoder.copy_buffer_to_buffer(
                self.buffers.query_hits.buffer(),
                0,
                self.buffers.query_pack.buffer(),
                header_bytes as u64,
                self.buffers.query_hits.size(),
            );
            self.buffers.queries_readback.enqueue(
                device,
                &mut encoder,
                self.buffers.query_pack.buffer(),
                step,
            )
        };
        encoder.copy_buffer_to_buffer(
            self.buffers.event_count.buffer(),
            0,
            self.buffers.events_pack.buffer(),
            0,
            12,
        );
        encoder.copy_buffer_to_buffer(
            self.buffers.events.buffer(),
            0,
            self.buffers.events_pack.buffer(),
            12,
            self.buffers.events.size(),
        );
        let stale_events = self.buffers.events_readback.enqueue(
            device,
            &mut encoder,
            self.buffers.events_pack.buffer(),
            step,
        );
        queue.submit([encoder.finish()]);
        self.buffers.bodies_readback.arm();
        self.buffers.queries_readback.arm();
        self.buffers.events_readback.arm();
        if let Some((stale_step, bytes)) = stale_bodies {
            self.consume_bodies(stale_step, &bytes);
        }
        if let Some((stale_step, bytes)) = stale_queries {
            self.consume_queries(stale_step, &bytes);
        }
        if let Some((stale_step, bytes)) = stale_events {
            self.consume_events(stale_step, &bytes);
        }
        self.commands.clear();
        self.constraint_commands.clear();
        self.queries.clear();
        self.step_index += 1;
    }

    pub fn poll(&mut self) {
        self.collect_readbacks();
    }

    pub fn wait(&mut self) {
        self.gpu
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device lost while awaiting readback");
        self.collect_readbacks();
    }

    pub fn wake(&mut self, handle: BodyHandle) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::wake(slot));
    }

    pub fn sleep(&mut self, handle: BodyHandle) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::sleep(slot));
    }

    pub fn read_state(&self, handle: BodyHandle) -> BodyState {
        self.validate(handle);
        self.states[handle.id as usize].expect("body state is unavailable")
    }

    fn island_rounds(&self) -> u32 {
        (self.capacity as u32).ilog2() + 1
    }

    fn inverse_mass_of(&self, desc: &BodyDesc) -> f32 {
        if desc.mass > 0.0 { 1.0 / desc.mass } else { 0.0 }
    }

    fn description(&self, handle: BodyHandle) -> BodyDesc {
        let id = handle.id as usize;
        BodyDesc {
            colliders: self.collider_descs[id].clone(),
            position: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            mass: self.masses[id],
            collision_group: 0,
            collision_mask: 0,
            kinematic: false,
            ccd: false,
        }
    }

    fn command_slot(&self, handle: BodyHandle) -> u32 {
        self.validate(handle);
        self.index_of[handle.id as usize]
    }

    fn schedule_patch(&mut self, handle: BodyHandle, mask: u32, record: RigidBodyRecord) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::patch(
            slot,
            mask,
            record,
            [ColliderRecord::zeroed(); 4],
            0,
        ));
    }

    fn schedule_patch_collider(
        &mut self,
        handle: BodyHandle,
        mask: u32,
        record: RigidBodyRecord,
        collider: ColliderRecord,
        index: u32,
    ) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::patch(
            slot,
            mask,
            record,
            [collider, ColliderRecord::zeroed(), ColliderRecord::zeroed(), ColliderRecord::zeroed()],
            index,
        ));
    }

    fn validate(&self, handle: BodyHandle) {
        let id = handle.id as usize;
        if id >= self.capacity {
            panic!("body handle {handle:?} is out of range");
        }
        if self.generations[id] != handle.generation {
            panic!("body handle {handle:?} is stale");
        }
        if self.index_of[id] == u32::MAX {
            panic!("body handle {handle:?} is not alive");
        }
    }

    fn validate_constraint(&self, handle: ConstraintHandle) {
        let id = handle.id as usize;
        if id >= self.constraint_capacity {
            panic!("constraint handle {handle:?} is out of range");
        }
        if self.constraint_generations[id] != handle.generation {
            panic!("constraint handle {handle:?} is stale");
        }
        if self.constraint_index_of[id] == u32::MAX {
            panic!("constraint handle {handle:?} is not alive");
        }
    }

    fn validate_query(&self, handle: QueryHandle) {
        let slot = handle.slot as usize;
        assert!(
            slot < self.query_pool.capacity(),
            "query handle {handle:?} is out of range"
        );
        assert_eq!(
            self.query_pool.generation(slot),
            handle.generation,
            "query handle {handle:?} is stale"
        );
    }

    fn assert_no_constraints(&self, handle: BodyHandle) {
        for constraint in &self.constraint_alive {
            if constraint.id == handle.id {
                panic!(
                    "body handle {handle:?} is referenced by a live constraint; remove it first"
                );
            }
        }
    }

    fn assert_unit(&self, orientation: [f32; 4]) {
        assert!(
            (orientation[0] * orientation[0]
                + orientation[1] * orientation[1]
                + orientation[2] * orientation[2]
                + orientation[3] * orientation[3]
                - 1.0)
                .abs()
                < 1e-4,
            "orientation must be a unit quaternion"
        );
    }

    fn collect_readbacks(&mut self) {
        for (step, bytes) in self.buffers.bodies_readback.poll(self.gpu.device()) {
            self.consume_bodies(step, &bytes);
        }
        for (step, bytes) in self.buffers.queries_readback.poll(self.gpu.device()) {
            self.consume_queries(step, &bytes);
        }
        while let Some((step, bytes)) = self.pending_events.pop_front() {
            self.consume_events(step, &bytes);
        }
    }

    fn consume_queries(&mut self, step: u64, bytes: &[u8]) {
        self.query_pool.consume(step, bytes);
    }

    fn consume_events(&mut self, step: u64, bytes: &[u8]) {
        let event_count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        let events_bytes = &bytes[12..12 + self.event_capacity * std::mem::size_of::<dynamis_layout::ContactEventRecord>()];
        let records: &[dynamis_layout::ContactEventRecord] = bytemuck::cast_slice(events_bytes);
        let count = event_count.min(self.event_capacity);
        for record in &records[..count] {
            if record.kind == dynamis_layout::EVENT_BEGIN
                || record.kind == dynamis_layout::EVENT_END
            {
                if record.first_id == u32::MAX || record.first_generation == 0 {
                    continue;
                }
                self.events.push(ContactEvent {
                    kind: if record.kind == dynamis_layout::EVENT_BEGIN {
                        ContactEventKind::Begin
                    } else {
                        ContactEventKind::End
                    },
                    first: BodyHandle {
                        id: record.first_id,
                        generation: record.first_generation,
                    },
                    second: BodyHandle {
                        id: record.second_id,
                        generation: record.second_generation,
                    },
                    sensor: record.sensor == 1,
                    point: record.point,
                    normal: record.normal,
                });
            }
        }
        let _ = step;
    }

    fn consume_bodies(&mut self, step: u64, bytes: &[u8]) {
        let records: &[RigidBodyRecord] = bytemuck::cast_slice(bytes);
        for record in records {
            let id = record.body_id as usize;
            assert!(
                id < self.capacity,
                "GPU readback returned an out-of-range body id"
            );
            if record.generation != self.generations[id] {
                continue;
            }
            if self.index_of[id] == u32::MAX {
                continue;
            }
            if self.states.len() <= id {
                self.states.resize(id + 1, None);
            }
            self.states[id] = Some(BodyState {
                position: record.position,
                orientation: record.orientation,
                velocity: record.velocity,
                angular_velocity: record.angular_velocity,
                inverse_mass: record.inverse_mass,
                sleeping: record.flags & BODY_SLEEPING != 0,
                step,
            });
        }
    }
}

fn layout_inverse_mass(desc: &BodyDesc) -> f32 {
    if desc.mass > 0.0 { 1.0 / desc.mass } else { 0.0 }
}
