mod body;
mod constraint;
mod event;
mod query;
mod readback;
mod shape;
mod step;

use crate::buffers::WorldBuffers;
use crate::pipeline::Pipeline;
use crate::query_pool::QueryPool;
use crate::shape_pool::ShapePool;
use crate::static_aabb;
use dynamis_gpu::GpuContext;
use dynamis_layout::{
    AabbRecord, BODY_SLEEPING, BodyCommandRecord, ColliderRecord, ConstraintCommandRecord,
    ConstraintRecord, Counter, QueryRecord, QueryResultHeader, RigidBodyRecord,
};
use dynamis_model::{
    BodyDesc, BodyHandle, BodyState, ColliderDesc, ConstraintHandle, ContactEvent,
    MAX_COLLIDERS_PER_BODY, MassProperties, PhysicsConfig, Shape,
};

pub(crate) const PAIR_CAPACITY_PER_BODY: usize = 64;
pub(crate) const QUERY_RATIO: usize = 2;

pub struct ContactPoint {
    pub position: [f32; 3],
    pub depth: f32,
    pub normal_impulse: f32,
    pub tangent_impulse: f32,
}

pub struct ContactManifold {
    pub first: BodyHandle,
    pub second: BodyHandle,
    pub sensor: bool,
    pub normal: [f32; 3],
    pub points: Vec<ContactPoint>,
    pub step: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct BodyDynamics {
    pub linear_damping: Option<f32>,
    pub angular_damping: Option<f32>,
    pub gravity_scale: f32,
    pub sleep_velocity: Option<f32>,
    pub sleep_angular_velocity: Option<f32>,
}

impl BodyDynamics {
    pub fn from_desc(desc: &BodyDesc) -> Self {
        Self {
            linear_damping: desc.linear_damping,
            angular_damping: desc.angular_damping,
            gravity_scale: desc.gravity_scale,
            sleep_velocity: desc.sleep_velocity,
            sleep_angular_velocity: desc.sleep_angular_velocity,
        }
    }

    pub fn defaults() -> Self {
        Self {
            linear_damping: None,
            angular_damping: None,
            gravity_scale: 1.0,
            sleep_velocity: None,
            sleep_angular_velocity: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugBuffer {
    Aabbs,
    LargeBodies,
    LargeCount,
    EntryCount,
    EntryKeys,
    EntryLo,
    PairCount,
    PairLo,
    PairHi,
    ContactValid,
    ContactCount,
    Contacts,
    PrevContacts,
    CompactRanks,
    CompactSums,
    CompactOffsets,
    ContactABodies,
    ConstraintKeysA,
    ConstraintValuesA,
    ConstraintFirstA,
    ConstraintKeysBOut,
    ConstraintValuesB,
    ConstraintFirstB,
    ConstraintDeltas,
    ConstraintCommandCount,
    Constraints,
    Events,
    EventCount,
    Overflow,
    IslandParents,
    IslandState,
    WakeFlags,
    QueryHeaders,
    QueryHits,
}

pub struct Simulation {
    gpu: GpuContext,
    config: PhysicsConfig,
    capacity: usize,
    reserved: usize,
    dynamic_count: usize,
    constraint_capacity: usize,
    step_index: u64,
    alive: Vec<BodyHandle>,
    index_of: Vec<u32>,
    generations: Vec<u32>,
    free_ids: Vec<u32>,
    commands: Vec<BodyCommandRecord>,
    collider_descs: Vec<Vec<ColliderDesc>>,
    masses: Vec<f32>,
    com_overrides: Vec<Option<[f32; 3]>>,
    inertia_overrides: Vec<Option<[f32; 6]>>,
    dynamics: Vec<BodyDynamics>,
    constraint_alive: Vec<ConstraintHandle>,
    constraint_index_of: Vec<u32>,
    constraint_generations: Vec<u32>,
    constraint_free_ids: Vec<u32>,
    constraint_commands: Vec<ConstraintCommandRecord>,
    queries: Vec<QueryRecord>,
    query_pool: QueryPool,
    shape_pool: ShapePool,
    events: Vec<ContactEvent>,
    event_sink: Option<Box<dyn FnMut(ContactEvent)>>,
    event_capacity: usize,
    broken_constraints: Vec<ConstraintHandle>,
    accumulator: f32,
    time_scale: f32,
    sub_dt: f32,
    buffers: WorldBuffers,
    pipeline: Pipeline,
    states: Vec<Option<BodyState>>,
    kinematic: Vec<bool>,
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
        let buffers = WorldBuffers::new(
            gpu.device(),
            shape_sources,
            capacity,
            pair_capacity,
            query_capacity,
            constraint_capacity,
        );
        buffers
            .prev_contact_count
            .write(gpu.queue(), bytemuck::cast_slice(&[Counter::none()]));
        let pipeline = Pipeline::new(&gpu, &buffers, capacity as u32);
        let query_header_bytes = query_capacity * std::mem::size_of::<QueryResultHeader>();
        let zeros = vec![0u8; query_header_bytes];
        buffers.query_headers.write(gpu.queue(), &zeros);
        Self {
            gpu,
            config,
            capacity,
            reserved: capacity,
            dynamic_count: 0,
            constraint_capacity,
            step_index: 0,
            alive: Vec::new(),
            index_of: vec![u32::MAX; capacity],
            generations: vec![1; capacity],
            free_ids: (0..capacity as u32).rev().collect(),
            commands: Vec::new(),
            collider_descs: vec![Vec::new(); capacity],
            masses: vec![1.0; capacity],
            com_overrides: vec![None; capacity],
            inertia_overrides: vec![None; capacity],
            dynamics: vec![BodyDynamics::defaults(); capacity],
            constraint_alive: Vec::new(),
            constraint_index_of: vec![u32::MAX; capacity],
            constraint_generations: vec![1; capacity],
            constraint_free_ids: (0..capacity as u32).rev().collect(),
            constraint_commands: Vec::new(),
            queries: Vec::new(),
            query_pool: QueryPool::new(query_capacity),
            shape_pool: ShapePool::new(shape_sources),
            events: Vec::new(),
            event_sink: None,
            event_capacity: pair_capacity,
            broken_constraints: Vec::new(),
            accumulator: 0.0,
            time_scale: 1.0,
            sub_dt: 1.0 / 60.0,
            buffers,
            pipeline,
            states: vec![None; capacity],
            kinematic: vec![false; capacity],
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
        if capacity <= self.capacity {
            return;
        }
        self.extend_free_ids(capacity);
        self.capacity = capacity;
        if capacity > self.reserved {
            let target = capacity.max(self.reserved.saturating_mul(2));
            self.rebuild(target);
        }
    }

    fn extend_free_ids(&mut self, capacity: usize) {
        let base = self.capacity;
        if capacity > base {
            self.free_ids
                .extend((base..capacity).rev().map(|id| id as u32));
        }
    }

    fn rebuild(&mut self, reserved: usize) {
        assert!(
            reserved >= self.alive.len(),
            "rebuild capacity must not drop below the live body count"
        );
        self.ensure_storage(reserved);
        let pair_capacity = reserved * PAIR_CAPACITY_PER_BODY;
        let query_capacity = reserved * QUERY_RATIO;
        let buffers = WorldBuffers::new(
            self.gpu.device(),
            self.shape_pool.capacity(),
            reserved,
            pair_capacity,
            query_capacity,
            self.constraint_capacity,
        );
        buffers
            .prev_contact_count
            .write(self.gpu.queue(), bytemuck::cast_slice(&[Counter::none()]));
        self.shape_pool.reset_upload_cursor();
        self.shape_pool.upload_pending(
            self.gpu.queue(),
            &buffers.shapes,
            &buffers.shape_vertices,
            &buffers.shape_triangles,
            &buffers.shape_nodes,
        );
        self.sync_state_to(&buffers);
        let pipeline = Pipeline::new(&self.gpu, &buffers, reserved as u32);
        self.buffers = buffers;
        self.pipeline = pipeline;
        self.reserved = reserved;
    }

    fn ensure_storage(&mut self, reserved: usize) {
        self.index_of.resize(reserved, u32::MAX);
        self.generations.resize(reserved, 1);
        self.collider_descs.resize(reserved, Vec::new());
        self.masses.resize(reserved, 1.0);
        self.com_overrides.resize(reserved, None);
        self.inertia_overrides.resize(reserved, None);
        self.dynamics.resize(reserved, BodyDynamics::defaults());
        self.states.resize(reserved, None);
        self.kinematic.resize(reserved, false);
    }

    fn sync_state_to(&mut self, buffers: &WorldBuffers) {
        let mut bodies = Vec::with_capacity(self.alive.len());
        let mut colliders = Vec::new();
        let mut aabbs = Vec::with_capacity(self.alive.len() * 4);
        for handle in self.alive.iter() {
            let id = handle.id as usize;
            let desc = self.build_desc(id);
            let mass = self.mass_properties_of(id);
            let mut body_record =
                RigidBodyRecord::build(&desc, handle.id, handle.generation, mass, &self.config);
            if let Some(state) = self.state_snapshot(id) {
                body_record.position = state.position;
                body_record.prev_position = state.prev_position;
                body_record.orientation = state.orientation;
                body_record.velocity = state.velocity;
                body_record.angular_velocity = state.angular_velocity;
                body_record.flags |= if state.sleeping { BODY_SLEEPING } else { 0 };
            }
            bodies.push(body_record);
            let block = self.collider_block_of(id);
            colliders.extend_from_slice(&block);
            let aabb_block = if !desc.kinematic && desc.mass <= 0.0 {
                static_aabb::static_aabbs(&body_record, &block, &self.shape_pool)
            } else {
                [AabbRecord::empty(); MAX_COLLIDERS_PER_BODY]
            };
            aabbs.extend_from_slice(&aabb_block);
        }
        buffers
            .bodies
            .write(self.gpu.queue(), bytemuck::cast_slice(&bodies));
        buffers
            .colliders
            .write(self.gpu.queue(), bytemuck::cast_slice(&colliders));
        buffers
            .aabbs
            .write(self.gpu.queue(), bytemuck::cast_slice(&aabbs));
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
            bytemuck::cast_slice(&[Counter::sized(constraint_count)]),
        );
    }

    fn build_desc(&self, id: usize) -> BodyDesc {
        let descs = &self.collider_descs[id];
        let dynamics = self.dynamics[id];
        BodyDesc {
            colliders: descs.clone(),
            position: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            mass: self.masses[id],
            density: None,
            com: self.com_overrides[id],
            inertia: self.inertia_overrides[id],
            collision_group: 0,
            collision_mask: 0,
            linear_damping: dynamics.linear_damping,
            angular_damping: dynamics.angular_damping,
            gravity_scale: dynamics.gravity_scale,
            sleep_velocity: dynamics.sleep_velocity,
            sleep_angular_velocity: dynamics.sleep_angular_velocity,
            kinematic: false,
            ccd: false,
        }
    }

    fn state_snapshot(&self, id: usize) -> Option<BodyState> {
        self.states[id]
    }

    fn mass_properties_of(&self, id: usize) -> MassProperties {
        self.build_desc(id)
            .mass_properties(|shape| self.shape_bounds(shape))
    }

    fn collider_record(&self, desc: &ColliderDesc) -> ColliderRecord {
        let source = match desc.shape {
            Shape::Hull(handle) | Shape::Mesh(handle) | Shape::HeightField(handle) => handle.id,
            _ => 0,
        };
        ColliderRecord::build(desc, source)
    }

    pub fn bodies_buffer(&self) -> &wgpu::Buffer {
        self.buffers.bodies.buffer()
    }

    pub fn colliders_buffer(&self) -> &wgpu::Buffer {
        self.buffers.colliders.buffer()
    }

    pub fn commands_buffer(&self) -> &wgpu::Buffer {
        self.buffers.commands.buffer()
    }

    pub fn debug_buffer(&self, which: DebugBuffer) -> &wgpu::Buffer {
        let buffer = match which {
            DebugBuffer::Aabbs => &self.buffers.aabbs,
            DebugBuffer::LargeBodies => &self.buffers.large_bodies,
            DebugBuffer::LargeCount => &self.buffers.large_count,
            DebugBuffer::EntryCount => &self.buffers.entry_count,
            DebugBuffer::EntryKeys => &self.buffers.entries.keys_hi,
            DebugBuffer::EntryLo => &self.buffers.entries.keys_lo,
            DebugBuffer::PairCount => &self.buffers.pair_count,
            DebugBuffer::PairLo => &self.buffers.pairs.keys_lo,
            DebugBuffer::PairHi => &self.buffers.pairs.keys_hi,
            DebugBuffer::ContactValid => &self.buffers.contact_valid,
            DebugBuffer::ContactCount => &self.buffers.contact_count,
            DebugBuffer::Contacts => &self.buffers.contacts,
            DebugBuffer::PrevContacts => &self.buffers.prev_contacts,
            DebugBuffer::CompactRanks => &self.buffers.compact_ranks,
            DebugBuffer::CompactSums => &self.buffers.compact_block_sums,
            DebugBuffer::CompactOffsets => &self.buffers.compact_block_offsets,
            DebugBuffer::ContactABodies => &self.buffers.contact_a_body,
            DebugBuffer::ConstraintKeysA => &self.buffers.constraint_gather_a_keys_out,
            DebugBuffer::ConstraintValuesA => &self.buffers.constraint_gather_a_values_out,
            DebugBuffer::ConstraintFirstA => &self.buffers.constraint_first_a,
            DebugBuffer::ConstraintKeysBOut => &self.buffers.constraint_gather_b_keys_out,
            DebugBuffer::ConstraintValuesB => &self.buffers.constraint_gather_b_values_out,
            DebugBuffer::ConstraintFirstB => &self.buffers.constraint_first_b,
            DebugBuffer::ConstraintDeltas => &self.buffers.constraint_deltas,
            DebugBuffer::ConstraintCommandCount => &self.buffers.constraint_command_count,
            DebugBuffer::Constraints => &self.buffers.constraints,
            DebugBuffer::Events => &self.buffers.events,
            DebugBuffer::EventCount => &self.buffers.event_count,
            DebugBuffer::Overflow => &self.buffers.overflow_flags,
            DebugBuffer::IslandParents => &self.buffers.island_parents,
            DebugBuffer::IslandState => &self.buffers.island_state,
            DebugBuffer::WakeFlags => &self.buffers.wake_flags,
            DebugBuffer::QueryHeaders => &self.buffers.query_headers,
            DebugBuffer::QueryHits => &self.buffers.query_hits,
        };
        buffer.buffer()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.gpu.queue()
    }

    pub fn device(&self) -> &wgpu::Device {
        self.gpu.device()
    }
}
