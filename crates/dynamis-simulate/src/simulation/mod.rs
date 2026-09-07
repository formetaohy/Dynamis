mod body;
mod constraint;
mod event;
mod query;
mod shape;

use crate::buffer::StageBuffers;
use crate::shape_pool::ShapePool;
use crate::stage::{Stages, build_stages, encode_physics};
use bytemuck::Zeroable;
use dynamis_gpu::GpuContext;
use dynamis_layout::{
    BODY_SLEEPING, BodyCommandRecord, ColliderRecord, ConstraintCommandRecord, ConstraintRecord,
    DispatchArgs, QueryRecord, QueryResultHeader, RigidBodyRecord, SimParamsRecord,
};
use dynamis_model::{
    BodyDesc, BodyHandle, BodyState, ColliderDesc, ConstraintHandle, ContactEvent, PhysicsConfig,
    Shape, inverse_inertia_diagonal,
};
use dynamis_query::QueryPool;
use std::collections::VecDeque;

pub(crate) const PAIR_CAPACITY_PER_BODY: usize = 64;
pub(crate) const QUERY_RATIO: usize = 2;

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
    PairArgs,
    ContactValid,
    ContactCount,
    Contacts,
    PrevContacts,
    CompactRanks,
    CompactSums,
    CompactSumsPad,
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
}

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
        let stages = build_stages(&gpu, &buffers, capacity as u32);
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
        buffers.prev_contact_count.write(
            self.gpu.queue(),
            bytemuck::cast_slice(&[DispatchArgs::none()]),
        );
        self.shape_pool.reset_upload_cursor();
        self.shape_pool.upload_pending(
            self.gpu.queue(),
            &buffers.shapes,
            &buffers.shape_vertices,
            &buffers.shape_triangles,
            &buffers.shape_nodes,
        );
        self.sync_state_to(&buffers);
        let stages = build_stages(&self.gpu, &buffers, capacity as u32);
        self.buffers = buffers;
        self.stages = stages;
        self.capacity = capacity;
    }

    fn sync_state_to(&mut self, buffers: &StageBuffers) {
        let mut bodies = Vec::with_capacity(self.alive.len());
        let mut colliders = Vec::new();
        for handle in self.alive.iter() {
            let id = handle.id as usize;
            let desc = self.build_desc(id);
            let inertia = self.compound_solid_inertia(&desc);
            let mut body_record =
                RigidBodyRecord::build(&desc, handle.id, handle.generation, inertia);
            if let Some(state) = self.state_snapshot(id) {
                body_record.position = state.position;
                body_record.prev_position = state.position;
                body_record.orientation = state.orientation;
                body_record.velocity = state.velocity;
                body_record.angular_velocity = state.angular_velocity;
                body_record.flags |= if state.sleeping { BODY_SLEEPING } else { 0 };
            }
            bodies.push(body_record);
            for index in 0..dynamis_model::MAX_COLLIDERS_PER_BODY {
                let collider = self
                    .collider_descs
                    .get(id)
                    .and_then(|descs| descs.get(index))
                    .map(|desc| self.collider_record(desc))
                    .unwrap_or_else(ColliderRecord::zeroed);
                colliders.push(collider);
            }
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
        let inverse_mass = desc.inverse_mass();
        if inverse_mass == 0.0 {
            return [0.0; 3];
        }
        let solid = desc
            .colliders
            .iter()
            .filter(|collider| !collider.sensor)
            .collect::<Vec<_>>();
        if solid.is_empty() {
            return [0.0; 3];
        }
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

    pub fn bodies_buffer(&self) -> &wgpu::Buffer {
        self.buffers.bodies_current.buffer()
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
            DebugBuffer::PairArgs => &self.buffers.pair_args,
            DebugBuffer::ContactValid => &self.buffers.contact_valid,
            DebugBuffer::ContactCount => &self.buffers.contact_count,
            DebugBuffer::Contacts => &self.buffers.contacts,
            DebugBuffer::PrevContacts => &self.buffers.prev_contacts,
            DebugBuffer::CompactRanks => &self.buffers.compact_ranks,
            DebugBuffer::CompactSums => &self.buffers.compact_block_sums,
            DebugBuffer::CompactSumsPad => &self.buffers.compact_block_sums_pad,
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
        };
        buffer.buffer()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.gpu.queue()
    }

    pub fn device(&self) -> &wgpu::Device {
        self.gpu.device()
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
            let header_bytes =
                self.query_pool.capacity() * std::mem::size_of::<QueryResultHeader>();
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

    fn island_rounds(&self) -> u32 {
        (self.capacity as u32).ilog2() + 1
    }
}
