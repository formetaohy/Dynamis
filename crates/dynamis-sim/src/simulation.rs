use crate::buffers::StageBuffers;
use crate::stages::{Stages, build_stages, encode_physics};
use bytemuck::Zeroable;
use dynamis_gpu::GpuContext;
use dynamis_layout::{
    BodyCommandRecord, ColliderRecord, ConstraintCommandRecord, ConstraintRecord, DispatchArgs,
    PATCH_COLLIDER, PATCH_FRICTION, PATCH_GROUP, PATCH_INVERSE_MASS, PATCH_KINEMATIC, PATCH_MASK,
    PATCH_ORIENTATION, PATCH_POSITION, PATCH_RESTITUTION, PATCH_ANGULAR_VELOCITY, PATCH_VELOCITY,
    QueryRecord, QueryResultRecord, RigidBodyRecord, SimParamsRecord,
};
use dynamis_model::{
    BodyDesc, BodyHandle, BodyState, ConstraintDesc, ConstraintHandle, PhysicsConfig, ShapeDesc,
};
use dynamis_query::{QueryHandle, QueryHit, QueryPool};

const PAIR_CAPACITY_PER_BODY: usize = 64;

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
    constraint_alive: Vec<ConstraintHandle>,
    constraint_index_of: Vec<u32>,
    constraint_generations: Vec<u32>,
    constraint_free_ids: Vec<u32>,
    constraint_commands: Vec<ConstraintCommandRecord>,
    queries: Vec<QueryRecord>,
    query_pool: QueryPool,
    buffers: StageBuffers,
    stages: Stages,
    states: Vec<Option<BodyState>>,
    shapes: Vec<ShapeDesc>,
    masses: Vec<f32>,
    kinematic_flags: Vec<bool>,
}

impl Simulation {
    pub fn new(gpu: GpuContext, capacity: usize, config: PhysicsConfig) -> Self {
        assert!(capacity > 0, "simulation capacity must be positive");
        let pair_capacity = capacity * PAIR_CAPACITY_PER_BODY;
        let query_capacity = capacity * 2;
        let constraint_capacity = capacity;
        let buffers = StageBuffers::new(
            gpu.device(),
            capacity,
            pair_capacity,
            query_capacity,
            constraint_capacity,
        );
        let stages = build_stages(gpu.device(), &buffers);
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
            constraint_alive: Vec::new(),
            constraint_index_of: vec![u32::MAX; capacity],
            constraint_generations: vec![1; capacity],
            constraint_free_ids: (0..capacity as u32).rev().collect(),
            constraint_commands: Vec::new(),
            queries: Vec::new(),
            query_pool: QueryPool::new(query_capacity),
            buffers,
            stages,
            states: vec![None; capacity],
            shapes: vec![ShapeDesc::sphere(1.0); capacity],
            masses: vec![1.0; capacity],
            kinematic_flags: vec![false; capacity],
        }
    }

    pub fn config(&self) -> &PhysicsConfig {
        &self.config
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

    pub fn bodies_buffer(&self) -> &wgpu::Buffer {
        self.buffers.bodies_current.buffer()
    }

    pub fn colliders_buffer(&self) -> &wgpu::Buffer {
        self.buffers.colliders.buffer()
    }

    pub fn commands_buffer(&self) -> &wgpu::Buffer {
        self.buffers.commands.buffer()
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

    pub fn debug_contacts(&self) -> &wgpu::Buffer {
        self.buffers.contacts.buffer()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.gpu.queue()
    }

    pub fn device(&self) -> &wgpu::Device {
        self.gpu.device()
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
        let body = RigidBodyRecord::build(&desc, handle.id, handle.generation);
        let collider = ColliderRecord::build(&desc);
        self.commands
            .push(BodyCommandRecord::add(slot, body, collider));
        self.states[id as usize] = Some(BodyState {
            position: desc.position,
            orientation: desc.orientation,
            velocity: desc.velocity,
            angular_velocity: desc.angular_velocity,
            inverse_mass: body.inverse_mass,
            step: self.step_index,
        });
        self.shapes[id as usize] = desc.shape;
        self.masses[id as usize] = desc.mass;
        self.kinematic_flags[id as usize] = desc.kinematic;
        handle
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
        self.states[id] = None;
        self.commands
            .push(BodyCommandRecord::remove(slot as u32, tail as u32));
    }

    pub fn set_position(&mut self, handle: BodyHandle, position: [f32; 3]) {
        let mut record = RigidBodyRecord::zeroed();
        record.position = position;
        self.schedule_patch(handle, PATCH_POSITION, record, None);
    }

    pub fn set_orientation(&mut self, handle: BodyHandle, orientation: [f32; 4]) {
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
        let mut record = RigidBodyRecord::zeroed();
        record.orientation = orientation;
        self.schedule_patch(handle, PATCH_ORIENTATION, record, None);
    }

    pub fn set_velocity(&mut self, handle: BodyHandle, velocity: [f32; 3]) {
        let mut record = RigidBodyRecord::zeroed();
        record.velocity = velocity;
        self.schedule_patch(handle, PATCH_VELOCITY, record, None);
    }

    pub fn set_angular_velocity(&mut self, handle: BodyHandle, angular_velocity: [f32; 3]) {
        let mut record = RigidBodyRecord::zeroed();
        record.angular_velocity = angular_velocity;
        self.schedule_patch(handle, PATCH_ANGULAR_VELOCITY, record, None);
    }

    pub fn set_mass(&mut self, handle: BodyHandle, mass: f32) {
        assert!(mass >= 0.0, "mass must be non-negative");
        self.validate(handle);
        self.masses[handle.id as usize] = mass;
        let mut record = RigidBodyRecord::zeroed();
        record.inverse_mass = inverse_mass(mass, self.is_kinematic(handle));
        record.inverse_inertia_body = self
            .shapes[handle.id as usize]
            .inverse_inertia_diagonal(record.inverse_mass);
        self.schedule_patch(handle, PATCH_INVERSE_MASS, record, None);
    }

    pub fn set_shape(&mut self, handle: BodyHandle, shape: ShapeDesc) {
        self.validate(handle);
        self.shapes[handle.id as usize] = shape;
        let mut record = RigidBodyRecord::zeroed();
        record.inverse_mass = self.states[handle.id as usize]
            .map(|state| state.inverse_mass)
            .unwrap_or(0.0);
        record.inverse_inertia_body = shape.inverse_inertia_diagonal(record.inverse_mass);
        let collider = ColliderRecord::build(&BodyDesc::new(shape));
        self.schedule_patch(handle, PATCH_COLLIDER, record, Some(collider));
    }

    pub fn set_restitution(&mut self, handle: BodyHandle, restitution: f32) {
        let mut record = RigidBodyRecord::zeroed();
        record.restitution = restitution;
        self.schedule_patch(handle, PATCH_RESTITUTION, record, None);
    }

    pub fn set_friction(&mut self, handle: BodyHandle, friction: f32) {
        assert!(friction >= 0.0, "friction must be non-negative");
        let mut record = RigidBodyRecord::zeroed();
        record.friction = friction;
        self.schedule_patch(handle, PATCH_FRICTION, record, None);
    }

    pub fn set_collision_group(&mut self, handle: BodyHandle, group: u32) {
        let mut record = RigidBodyRecord::zeroed();
        record.collision_group = group;
        self.schedule_patch(handle, PATCH_GROUP, record, None);
    }

    pub fn set_collision_mask(&mut self, handle: BodyHandle, mask: u32) {
        let mut record = RigidBodyRecord::zeroed();
        record.collision_mask = mask;
        self.schedule_patch(handle, PATCH_MASK, record, None);
    }

    pub fn set_kinematic(&mut self, handle: BodyHandle, kinematic: bool) {
        self.validate(handle);
        let mut record = RigidBodyRecord::zeroed();
        record.flags = if kinematic {
            dynamis_layout::BODY_KINEMATIC
        } else {
            0
        };
        record.inverse_mass = inverse_mass(self.masses[handle.id as usize], kinematic);
        record.inverse_inertia_body = self.shapes[handle.id as usize]
            .inverse_inertia_diagonal(record.inverse_mass);
        self.kinematic_flags[handle.id as usize] = kinematic;
        self.schedule_patch(handle, PATCH_KINEMATIC, record, None);
    }

    pub fn apply_force(&mut self, handle: BodyHandle, force: [f32; 3]) {
        let slot = self.command_slot(handle);
        self.commands.push(BodyCommandRecord::force(slot, force));
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
        if !matches!(desc.kind, dynamis_model::ConstraintKind::Ball | dynamis_model::ConstraintKind::Distance)
            && desc.axis == [0.0; 3]
        {
            panic!("constraint axis must be non-zero");
        }
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
        let record = ConstraintRecord::build(&desc, self.index_of[first.id as usize], self.index_of[second.id as usize]);
        self.constraint_commands
            .push(ConstraintCommandRecord::add(slot, record));
        handle
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
        self.constraint_commands
            .push(ConstraintCommandRecord::remove(slot as u32));
    }

    pub fn constraints(&self) -> &[ConstraintHandle] {
        &self.constraint_alive
    }

    pub fn raycast(&mut self, origin: [f32; 3], direction: [f32; 3], max_t: f32) -> QueryHandle {
        assert!(max_t > 0.0, "raycast distance must be positive");
        assert!(direction != [0.0; 3], "raycast direction must be non-zero");
        let slot = self.query_pool.allocate();
        let generation = self.query_pool.generation(slot);
        self.queries
            .push(QueryRecord::ray(origin, direction, max_t, slot as u32));
        QueryHandle {
            slot: slot as u32,
            generation,
        }
    }

    pub fn sphere_query(&mut self, center: [f32; 3], radius: f32) -> QueryHandle {
        assert!(radius > 0.0, "sphere query radius must be positive");
        let slot = self.query_pool.allocate();
        let generation = self.query_pool.generation(slot);
        self.queries
            .push(QueryRecord::sphere(center, radius, slot as u32));
        QueryHandle {
            slot: slot as u32,
            generation,
        }
    }

    pub fn query_hit(&self, handle: QueryHandle) -> Option<QueryHit> {
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
        self.query_pool.hit(slot)
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
            bytemuck::cast_slice(&[DispatchArgs::sized(
                self.constraint_commands.len() as u32,
            )]),
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
        if !self.queries.is_empty() {
            self.buffers
                .queries
                .write(queue, bytemuck::cast_slice(&self.queries));
            let slots = self.queries.iter().map(|query| query.slot).collect();
            self.query_pool.mark_batch(step, slots);
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dynamis step encoder"),
            });
        encode_physics(
            &self.stages,
            &self.buffers,
            device,
            queue,
            &mut encoder,
            self.alive.len() as u32,
            self.config.solve_iterations,
            self.config.position_iterations,
            self.queries.len() as u32,
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
            self.buffers.queries_readback.enqueue(
                device,
                &mut encoder,
                self.buffers.query_results.buffer(),
                step,
            )
        };
        queue.submit([encoder.finish()]);
        self.buffers.bodies_readback.arm();
        self.buffers.queries_readback.arm();
        if let Some((stale_step, bytes)) = stale_bodies {
            self.consume_bodies(stale_step, &bytes);
        }
        if let Some((stale_step, bytes)) = stale_queries {
            self.consume_queries(stale_step, &bytes);
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

    pub fn read_state(&self, handle: BodyHandle) -> BodyState {
        self.validate(handle);
        self.states[handle.id as usize].expect("body state is unavailable")
    }

    fn is_kinematic(&self, handle: BodyHandle) -> bool {
        self.validate(handle);
        self.kinematic_flags[handle.id as usize]
    }

    fn command_slot(&self, handle: BodyHandle) -> u32 {
        self.validate(handle);
        self.index_of[handle.id as usize]
    }

    fn schedule_patch(
        &mut self,
        handle: BodyHandle,
        mask: u32,
        record: RigidBodyRecord,
        collider: Option<ColliderRecord>,
    ) {
        let slot = self.command_slot(handle);
        let collider = collider.unwrap_or_else(ColliderRecord::zeroed);
        self.commands
            .push(BodyCommandRecord::patch(slot, mask, record, collider));
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

    fn assert_no_constraints(&self, handle: BodyHandle) {
        for constraint in &self.constraint_alive {
            if constraint.id == handle.id {
                panic!(
                    "body handle {handle:?} is referenced by a live constraint; remove it first"
                );
            }
        }
    }

    fn collect_readbacks(&mut self) {
        for (step, bytes) in self.buffers.bodies_readback.poll(self.gpu.device()) {
            self.consume_bodies(step, &bytes);
        }
        for (step, bytes) in self.buffers.queries_readback.poll(self.gpu.device()) {
            self.consume_queries(step, &bytes);
        }
    }

    fn consume_queries(&mut self, step: u64, bytes: &[u8]) {
        let records: &[QueryResultRecord] = bytemuck::cast_slice(bytes);
        self.query_pool.consume(step, records);
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
            self.states[id] = Some(BodyState {
                position: record.position,
                orientation: record.orientation,
                velocity: record.velocity,
                angular_velocity: record.angular_velocity,
                inverse_mass: record.inverse_mass,
                step,
            });
        }
    }
}

fn inverse_mass(mass: f32, kinematic: bool) -> f32 {
    if kinematic {
        return 0.0;
    }
    if mass > 0.0 {
        1.0 / mass
    } else {
        0.0
    }
}
