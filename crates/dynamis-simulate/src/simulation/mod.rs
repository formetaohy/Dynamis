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
use bytemuck::Zeroable;
use dynamis_gpu::GpuContext;
#[cfg(feature = "profile")]
use dynamis_gpu::GpuPassTiming;
use dynamis_layout::{
    AabbRecord, BodyCommandRecord, BodyDescriptorRecord, BodyStateRecord, ColliderRecord,
    ConstraintCommandRecord, ConstraintDescriptorRecord, ConstraintRuntimeRecord, Counter,
    QueryRecord, QueryResultHeader,
};
use dynamis_model::{
    BodyHandle, BodyState, ColliderDesc, ConstraintHandle, ContactEvent, MAX_COLLIDERS_PER_BODY,
    MassProperties, PhysicsConfig, Shape,
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
    BodyDescriptors,
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
    ConstraintDescriptors,
    ConstraintCommandCount,
    ConstraintRuntime,
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
    descriptors: Vec<BodyDescriptorRecord>,
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
    #[cfg(feature = "profile")]
    pass_timings: Vec<GpuPassTiming>,
    buffers: WorldBuffers,
    pipeline: Pipeline,
    states: Vec<Option<BodyState>>,
    kinematic: Vec<bool>,
    constraint_records: Vec<ConstraintDescriptorRecord>,
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
            descriptors: vec![BodyDescriptorRecord::zeroed(); capacity],
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
            #[cfg(feature = "profile")]
            pass_timings: Vec::new(),
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
        self.transfer_device_state(&buffers);
        self.upload_host_state(&buffers);
        self.shape_pool.reset_upload_cursor();
        self.shape_pool.upload_pending(
            self.gpu.queue(),
            &buffers.shapes,
            &buffers.shape_vertices,
            &buffers.shape_triangles,
            &buffers.shape_nodes,
        );
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
        self.descriptors
            .resize(reserved, BodyDescriptorRecord::zeroed());
        self.states.resize(reserved, None);
        self.kinematic.resize(reserved, false);
    }

    /// Device-owned rows survive a reallocation untouched; only their storage moves.
    fn transfer_device_state(&self, next: &WorldBuffers) {
        let previous = &self.buffers;
        let bodies = (self.alive.len() * std::mem::size_of::<BodyStateRecord>()) as u64;
        let aabbs =
            (self.alive.len() * MAX_COLLIDERS_PER_BODY * std::mem::size_of::<AabbRecord>()) as u64;
        let contacts = previous.prev_contacts.size().min(next.prev_contacts.size());
        let constraints =
            (self.constraint_alive.len() * std::mem::size_of::<ConstraintRuntimeRecord>()) as u64;
        let mut encoder =
            self.gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dynamis grow transfer"),
                });
        let mut copy =
            |source: &dynamis_gpu::GpuBuffer, target: &dynamis_gpu::GpuBuffer, bytes: u64| {
                encoder.copy_buffer_to_buffer(source.buffer(), 0, target.buffer(), 0, bytes);
            };
        copy(&previous.body_states, &next.body_states, bodies);
        copy(&previous.aabbs, &next.aabbs, aabbs);
        copy(&previous.prev_contacts, &next.prev_contacts, contacts);
        copy(
            &previous.prev_contact_count,
            &next.prev_contact_count,
            std::mem::size_of::<Counter>() as u64,
        );
        copy(
            &previous.constraint_runtime,
            &next.constraint_runtime,
            constraints,
        );
        self.gpu.queue().submit([encoder.finish()]);
    }

    /// Host-owned rows are re-emitted in slot order from the host authority.
    fn upload_host_state(&self, next: &WorldBuffers) {
        let queue = self.gpu.queue();
        let mut descriptors = Vec::with_capacity(self.alive.len());
        let mut colliders = Vec::with_capacity(self.alive.len() * MAX_COLLIDERS_PER_BODY);
        for handle in &self.alive {
            descriptors.push(self.descriptors[handle.id as usize]);
            colliders.extend_from_slice(&self.collider_block_of(handle.id as usize));
        }
        next.body_descs
            .write(queue, bytemuck::cast_slice(&descriptors));
        next.colliders
            .write(queue, bytemuck::cast_slice(&colliders));
        next.constraint_descs.write(
            queue,
            bytemuck::cast_slice(&self.constraint_records[..self.constraint_alive.len()]),
        );
        next.constraint_count_state.write(
            queue,
            bytemuck::cast_slice(&[Counter::sized(self.constraint_alive.len() as u32)]),
        );
    }

    fn state_snapshot(&self, id: usize) -> Option<BodyState> {
        self.states[id]
    }

    fn mass_properties_of(&self, id: usize) -> MassProperties {
        dynamis_model::mass_properties_of_intent(
            &self.collider_descs[id],
            self.masses[id],
            self.com_overrides[id],
            self.inertia_overrides[id],
            |shape| self.shape_bounds(shape),
        )
    }

    fn collider_record(&self, desc: &ColliderDesc) -> ColliderRecord {
        let source = match desc.shape {
            Shape::Hull(handle) | Shape::Mesh(handle) | Shape::HeightField(handle) => handle.id,
            _ => 0,
        };
        ColliderRecord::build(desc, source)
    }

    /// The device-owned kinematic state of every live body slot.
    pub fn body_states_buffer(&self) -> &wgpu::Buffer {
        self.buffers.body_states.buffer()
    }

    /// The host-owned invariant properties of every live body slot.
    pub fn body_descriptors_buffer(&self) -> &wgpu::Buffer {
        self.buffers.body_descs.buffer()
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
            DebugBuffer::BodyDescriptors => &self.buffers.body_descs,
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
            DebugBuffer::ConstraintDescriptors => &self.buffers.constraint_descs,
            DebugBuffer::ConstraintRuntime => &self.buffers.constraint_runtime,
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
