mod body;
mod constraint;
mod event;
mod pack;
mod query;
mod readback;
mod shape;
mod step;

use crate::StreamBudget;
use crate::buffers::WorldBuffers;
use crate::pipeline::Pipeline;
use crate::query_pool::QueryPool;
use crate::reservation::{Live, Reservation, ShapeReservation};
use crate::shape_pool::ShapePool;
use bytemuck::Zeroable;
use dynamis_gpu::GpuContext;
#[cfg(feature = "profile")]
use dynamis_gpu::GpuPassTiming;
use dynamis_layout::{
    BodyCommandRecord, BodyDescriptorRecord, COUNTER_BODIES, COUNTER_BODY_COMMANDS,
    COUNTER_CONSTRAINT_COMMANDS, COUNTER_CONSTRAINTS, COUNTER_PREV_CONTACTS, ColliderRecord,
    ConstraintCommandRecord, ConstraintDescriptorRecord, Counters, QueryResultRecord,
};
use dynamis_model::{
    BodyHandle, BodyState, ColliderDesc, ConstraintHandle, ContactEvent, MAX_COLLIDERS_PER_BODY,
    MassProperties, PhysicsConfig, Shape,
};

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
    EntryKeys,
    EntryLo,
    PairLo,
    PairHi,
    ContactValid,
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
    ConstraintKeysB,
    ConstraintValuesB,
    ConstraintFirstB,
    ConstraintDeltas,
    ConstraintDescriptors,
    ConstraintRuntime,
    Events,
    IslandParents,
    IslandState,
    WakeFlags,
    QueryResults,
    Counters,
}

pub struct Simulation {
    gpu: GpuContext,
    config: PhysicsConfig,
    /// The host-side handle space; bounds ids, never device work.
    slots: usize,
    reservation: Reservation,
    shape_reservation: ShapeReservation,
    /// What the device measured on the step whose counters last arrived.
    observed: Counters,
    dynamic_count: usize,
    device_body_count: u32,
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
    queries: Vec<dynamis_layout::QueryRecord>,
    next_batch: u64,
    query_pool: QueryPool,
    shape_pool: ShapePool,
    events: Vec<ContactEvent>,
    event_sink: Option<Box<dyn FnMut(ContactEvent)>>,
    broken_constraints: Vec<ConstraintHandle>,
    dirty_bodies: Vec<u32>,
    dirty_constraints: Vec<u32>,
    shapes_dirty: bool,
    stream_budget: StreamBudget,
    accumulator: f32,
    time_scale: f32,
    sub_dt: f32,
    #[cfg(feature = "profile")]
    pass_timings: Vec<GpuPassTiming>,
    buffers: WorldBuffers,
    pipeline: Pipeline,
    states: Vec<Option<BodyState>>,
    states_synchronized: bool,
    kinematic: Vec<bool>,
    constraint_records: Vec<ConstraintDescriptorRecord>,
}

impl Simulation {
    /// A world able to address `slots` body and constraint ids. Device storage is
    /// planned from what is live, so this bound costs nothing until it is used;
    /// `budget` sets how much the first plan gives each live body.
    pub fn new(gpu: GpuContext, slots: usize, config: PhysicsConfig, budget: StreamBudget) -> Self {
        assert!(slots > 0, "simulation slot count must be positive");
        assert!(
            u32::try_from(slots).is_ok(),
            "simulation slot count exceeds the handle space"
        );
        assert!(budget.pairs_per_body > 0, "pairs per body must be positive");
        assert!(
            budget.events_per_body > 0,
            "events per body must be positive"
        );
        let reservation = Reservation::initial(budget.pairs_per_body, budget.events_per_body);
        let shape_pool = ShapePool::new();
        let shape_reservation =
            ShapeReservation::planned(&ShapeReservation::EMPTY, &shape_pool.used());
        let buffers = WorldBuffers::new(gpu.device(), &reservation, &shape_reservation);
        let pipeline = Pipeline::new(&gpu, &buffers, &reservation);
        Self {
            gpu,
            config,
            slots,
            reservation,
            shape_reservation,
            observed: [0; dynamis_layout::COUNTER_COUNT],
            dynamic_count: 0,
            device_body_count: 0,
            step_index: 0,
            alive: Vec::new(),
            index_of: vec![u32::MAX; slots],
            generations: vec![1; slots],
            free_ids: (0..slots as u32).rev().collect(),
            commands: Vec::new(),
            collider_descs: vec![Vec::new(); slots],
            masses: vec![1.0; slots],
            com_overrides: vec![None; slots],
            inertia_overrides: vec![None; slots],
            descriptors: vec![BodyDescriptorRecord::zeroed(); slots],
            constraint_alive: Vec::new(),
            constraint_index_of: vec![u32::MAX; slots],
            constraint_generations: vec![1; slots],
            constraint_free_ids: (0..slots as u32).rev().collect(),
            constraint_commands: Vec::new(),
            queries: Vec::new(),
            next_batch: 0,
            query_pool: QueryPool::new(),
            shape_pool,
            events: Vec::new(),
            event_sink: None,
            broken_constraints: Vec::new(),
            dirty_bodies: Vec::new(),
            dirty_constraints: Vec::new(),
            shapes_dirty: false,
            stream_budget: budget,
            accumulator: 0.0,
            time_scale: 1.0,
            sub_dt: 1.0 / 60.0,
            #[cfg(feature = "profile")]
            pass_timings: Vec::new(),
            buffers,
            pipeline,
            states: vec![None; slots],
            states_synchronized: true,
            kinematic: vec![false; slots],
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

    /// The number of body ids this world can address.
    pub fn capacity(&self) -> usize {
        self.slots
    }

    pub fn bodies(&self) -> &[BodyHandle] {
        &self.alive
    }

    pub fn count(&self) -> usize {
        self.alive.len()
    }

    /// Raises the id space. Device storage follows live bodies, so this alone never
    /// reallocates anything on the card.
    pub fn grow(&mut self, slots: usize) {
        assert!(
            u32::try_from(slots).is_ok(),
            "simulation slot count exceeds the handle space"
        );
        assert!(
            slots >= self.alive.len(),
            "grow slot count must not drop below the live body count"
        );
        if slots <= self.slots {
            return;
        }
        let base = self.slots;
        self.slots = slots;
        self.free_ids
            .extend((base..slots).rev().map(|id| id as u32));
        self.ensure_storage();
    }

    fn ensure_storage(&mut self) {
        self.index_of.resize(self.slots, u32::MAX);
        self.generations.resize(self.slots, 1);
        self.collider_descs.resize(self.slots, Vec::new());
        self.masses.resize(self.slots, 1.0);
        self.com_overrides.resize(self.slots, None);
        self.inertia_overrides.resize(self.slots, None);
        self.descriptors
            .resize(self.slots, BodyDescriptorRecord::zeroed());
        self.states.resize(self.slots, None);
        self.kinematic.resize(self.slots, false);
        self.constraint_index_of.resize(self.slots, u32::MAX);
        self.constraint_generations.resize(self.slots, 1);
    }

    fn live(&self) -> Live {
        Live {
            bodies: self.alive.len() as u32,
            constraints: self.constraint_alive.len() as u32,
            body_commands: self.commands.len() as u32,
            constraint_commands: self.constraint_commands.len() as u32,
            queries: self.queries.len() as u32,
        }
    }

    /// Publishes every host-side row edit the world accumulated since the last step.
    ///
    /// Host mutations never touch the device directly, so this is the only place that
    /// writes rows and the only place that may need to reallocate first.
    pub(crate) fn flush_rows(&mut self) {
        let queue = self.gpu.queue();
        if self.shapes_dirty {
            self.shape_pool.upload_pending(
                queue,
                &self.buffers.shapes,
                &self.buffers.shape_vertices,
                &self.buffers.shape_triangles,
                &self.buffers.shape_nodes,
            );
            self.shapes_dirty = false;
        }
        self.dirty_bodies.sort_unstable();
        self.dirty_bodies.dedup();
        for slot in std::mem::take(&mut self.dirty_bodies) {
            let id = self.alive[slot as usize].id as usize;
            let colliders = self.collider_block_of(id);
            let descriptor = self.descriptors[id];
            let aabbs = self.aabb_block_of(id);
            self.buffers.colliders.write_at(
                queue,
                slot as u64 * self.buffers.collider_row(),
                bytemuck::cast_slice(&colliders),
            );
            self.buffers.body_descs.write_at(
                queue,
                slot as u64 * self.buffers.descriptor_row(),
                bytemuck::cast_slice(&[descriptor]),
            );
            self.buffers.aabbs.write_at(
                queue,
                slot as u64 * self.buffers.aabb_row(),
                bytemuck::cast_slice(&aabbs),
            );
        }
        self.dirty_constraints.sort_unstable();
        self.dirty_constraints.dedup();
        for slot in std::mem::take(&mut self.dirty_constraints) {
            let record = self.constraint_records[slot as usize];
            self.buffers.constraint_descs.write_at(
                queue,
                slot as u64 * self.buffers.constraint_row(),
                bytemuck::cast_slice(&[record]),
            );
        }
    }

    pub(crate) fn apply_plan(&mut self) {
        let next = Reservation::planned(
            &self.reservation,
            &self.live(),
            self.stream_budget.pairs_per_body,
            self.stream_budget.events_per_body,
        );
        let shapes = ShapeReservation::planned(&self.shape_reservation, &self.shape_pool.used());
        if next == self.reservation && shapes == self.shape_reservation {
            return;
        }
        self.reservation = next;
        self.shape_reservation = shapes;
        self.drain_in_flight();
        self.rebuild();
    }

    /// A reallocation destroys the buffers holding in-flight reads, so every pending
    /// result is brought home first.
    fn drain_in_flight(&mut self) {
        self.drain_readbacks();
    }

    pub(crate) fn rebuild(&mut self) {
        let buffers = WorldBuffers::new(
            self.gpu.device(),
            &self.reservation,
            &self.shape_reservation,
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

        self.pipeline = Pipeline::new(&self.gpu, &buffers, &self.reservation);
        self.buffers = buffers;
    }

    /// Device-owned rows survive a reallocation untouched; only their storage moves.
    fn transfer_device_state(&self, next: &WorldBuffers) {
        let previous = &self.buffers;
        let bodies = previous.body_states.size().min(next.body_states.size());
        let aabbs = previous.aabbs.size().min(next.aabbs.size());
        let contacts = previous.contacts.size().min(next.contacts.size());
        let constraints = previous
            .constraint_runtime
            .size()
            .min(next.constraint_runtime.size());
        let mut encoder =
            self.gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dynamis reallocate"),
                });
        let mut copy =
            |source: &dynamis_gpu::GpuBuffer, target: &dynamis_gpu::GpuBuffer, bytes: u64| {
                encoder.copy_buffer_to_buffer(source.buffer(), 0, target.buffer(), 0, bytes);
            };
        copy(&previous.body_states, &next.body_states, bodies);
        copy(&previous.aabbs, &next.aabbs, aabbs);
        copy(&previous.prev_contacts, &next.prev_contacts, contacts);
        copy(
            &previous.constraint_runtime,
            &next.constraint_runtime,
            constraints,
        );
        let (offset, width) = (
            COUNTER_PREV_CONTACTS as u64 * dynamis_layout::COUNTER_STRIDE,
            4,
        );
        encoder.copy_buffer_to_buffer(
            previous.counters.buffer(),
            offset,
            next.counters.buffer(),
            offset,
            width,
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
    }

    pub(crate) fn write_declared_counters(&self, queue: &wgpu::Queue) {
        let declared = [
            (COUNTER_BODIES, self.alive.len() as u32),
            (COUNTER_CONSTRAINTS, self.constraint_alive.len() as u32),
            (COUNTER_BODY_COMMANDS, self.commands.len() as u32),
            (
                COUNTER_CONSTRAINT_COMMANDS,
                self.constraint_commands.len() as u32,
            ),
        ];
        for (slot, value) in declared {
            self.buffers.counters.write_at(
                queue,
                slot as u64 * dynamis_layout::COUNTER_STRIDE,
                bytemuck::cast_slice(&[value]),
            );
        }
    }

    pub(crate) fn state_snapshot(&self, id: usize) -> Option<BodyState> {
        assert!(
            self.states_synchronized,
            "body states require synchronize_states() after stepping"
        );
        self.states[id]
    }

    pub(crate) fn mass_properties_of(&self, id: usize) -> MassProperties {
        dynamis_model::mass_properties_of_intent(
            &self.collider_descs[id],
            self.masses[id],
            self.com_overrides[id],
            self.inertia_overrides[id],
            |shape| self.shape_bounds(shape),
        )
    }

    pub(crate) fn collider_record(&self, desc: &ColliderDesc) -> ColliderRecord {
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
            DebugBuffer::EntryKeys => &self.buffers.entries.keys_hi,
            DebugBuffer::EntryLo => &self.buffers.entries.keys_lo,
            DebugBuffer::PairLo => &self.buffers.pairs.keys_lo,
            DebugBuffer::PairHi => &self.buffers.pairs.keys_hi,
            DebugBuffer::ContactValid => &self.buffers.contact_valid,
            DebugBuffer::Contacts => &self.buffers.contacts,
            DebugBuffer::PrevContacts => &self.buffers.prev_contacts,
            DebugBuffer::BodyDescriptors => &self.buffers.body_descs,
            DebugBuffer::CompactRanks => &self.buffers.compact_ranks,
            DebugBuffer::CompactSums => &self.buffers.compact_block_sums,
            DebugBuffer::CompactOffsets => &self.buffers.compact_block_offsets,
            DebugBuffer::ContactABodies => &self.buffers.contact_a_body,
            DebugBuffer::ConstraintKeysA => &self.buffers.constraint_a_keys,
            DebugBuffer::ConstraintValuesA => &self.buffers.constraint_a_values,
            DebugBuffer::ConstraintFirstA => &self.buffers.constraint_first_a,
            DebugBuffer::ConstraintKeysB => &self.buffers.constraint_b_keys,
            DebugBuffer::ConstraintValuesB => &self.buffers.constraint_b_values,
            DebugBuffer::ConstraintFirstB => &self.buffers.constraint_first_b,
            DebugBuffer::ConstraintDeltas => &self.buffers.constraint_deltas,
            DebugBuffer::ConstraintDescriptors => &self.buffers.constraint_descs,
            DebugBuffer::ConstraintRuntime => &self.buffers.constraint_runtime,
            DebugBuffer::Events => &self.buffers.events,
            DebugBuffer::IslandParents => &self.buffers.island_parents,
            DebugBuffer::IslandState => &self.buffers.island_state,
            DebugBuffer::WakeFlags => &self.buffers.wake_flags,
            DebugBuffer::QueryResults => &self.buffers.query_results,
            DebugBuffer::Counters => &self.buffers.counters,
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

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<QueryResultRecord>() == 784);
};
