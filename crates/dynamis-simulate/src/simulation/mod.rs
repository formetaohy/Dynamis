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
    BODY_SLEEPING, BodyCommandRecord, CONSTRAINT_INVALID, ColliderRecord, ConstraintCommandRecord,
    ConstraintRecord, ContactRecord, DispatchArgs, QueryRecord, QueryResultHeader, RigidBodyRecord,
    SimParamsRecord,
};
use dynamis_model::{
    BodyDesc, BodyHandle, BodyState, ColliderDesc, ConstraintHandle, ContactEvent, MassProperties,
    PhysicsConfig, Shape,
};
use dynamis_query::QueryPool;
use std::collections::VecDeque;

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
    pending_events: VecDeque<(u64, Vec<u8>)>,
    event_capacity: usize,
    broken_constraints: Vec<ConstraintHandle>,
    accumulator: f32,
    time_scale: f32,
    sub_dt: f32,
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
            com_overrides: Vec::new(),
            inertia_overrides: Vec::new(),
            dynamics: Vec::new(),
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
            broken_constraints: Vec::new(),
            accumulator: 0.0,
            time_scale: 1.0,
            sub_dt: 1.0 / 60.0,
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

    pub fn set_time_scale(&mut self, time_scale: f32) {
        assert!(time_scale > 0.0, "time scale must be strictly positive");
        self.time_scale = time_scale;
    }

    pub fn update(&mut self, real_dt: f32, sub_dt: f32, max_substeps: u32) {
        assert!(real_dt >= 0.0, "real dt must be non-negative");
        assert!(sub_dt > 0.0, "sub dt must be strictly positive");
        assert!(max_substeps > 0, "max substeps must be positive");
        self.sub_dt = sub_dt;
        self.accumulator += real_dt * self.time_scale;
        let mut steps = 0;
        while self.accumulator >= sub_dt && steps < max_substeps {
            self.step(sub_dt);
            self.accumulator -= sub_dt;
            steps += 1;
        }
        if steps == max_substeps {
            self.accumulator %= sub_dt;
        }
    }

    pub fn interpolation_alpha(&self) -> f32 {
        (self.accumulator / self.sub_dt).clamp(0.0, 1.0)
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
            let mass = self.mass_properties_of(id);
            let mut body_record =
                RigidBodyRecord::build(&desc, handle.id, handle.generation, mass, &self.config);
            if let Some(state) = self.state_snapshot(id) {
                body_record.position = state.position;
                body_record.prev_position = state.previous_position;
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
        self.states.get(id).cloned().flatten()
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
        let stale_constraints = self.buffers.constraints_readback.enqueue(
            device,
            &mut encoder,
            self.buffers.constraints.buffer(),
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
        self.buffers.constraints_readback.arm();
        if let Some((stale_step, bytes)) = stale_bodies {
            self.consume_bodies(stale_step, &bytes);
        }
        if let Some((stale_step, bytes)) = stale_constraints {
            self.consume_constraints(stale_step, &bytes);
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

    pub fn contact_manifolds(&mut self) -> Vec<ContactManifold> {
        self.wait();
        let count_bytes = std::mem::size_of::<DispatchArgs>();
        let args: Vec<DispatchArgs> = bytemuck::cast_slice(
            &self.read_gpu(self.buffers.contact_count.buffer(), count_bytes as u64),
        )
        .to_vec();
        let count = (args[0].x as usize).min(self.buffers.contact_capacity() as usize);
        if count == 0 {
            return Vec::new();
        }
        let record_bytes = (count * std::mem::size_of::<ContactRecord>()) as u64;
        let bytes = self.read_gpu(self.buffers.contacts.buffer(), record_bytes);
        let records: &[ContactRecord] = bytemuck::cast_slice(&bytes);
        records
            .iter()
            .map(|record| ContactManifold {
                first: BodyHandle {
                    id: record.first_body_id,
                    generation: record.first_generation,
                },
                second: BodyHandle {
                    id: record.second_body_id,
                    generation: record.second_generation,
                },
                sensor: record.sensor == 1,
                normal: record.normal,
                points: record.points[..record.point_count as usize]
                    .iter()
                    .map(|point| ContactPoint {
                        position: point.position,
                        depth: point.depth,
                        normal_impulse: point.accumulated_normal,
                        tangent_impulse: (point.accumulated_tangent_1
                            * point.accumulated_tangent_1
                            + point.accumulated_tangent_2 * point.accumulated_tangent_2)
                            .sqrt(),
                    })
                    .collect(),
                step: self.step_index.saturating_sub(1),
            })
            .collect()
    }

    pub fn overflow(&mut self) -> (u32, u32) {
        self.wait();
        let bytes = self.read_gpu(self.buffers.overflow_flags.buffer(), 8);
        (
            u32::from_le_bytes(bytes[0..4].try_into().expect("overflow read")),
            u32::from_le_bytes(bytes[4..8].try_into().expect("overflow read")),
        )
    }

    fn read_gpu(&self, buffer: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        let staging = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("dynamis sync readback"),
            size: bytes.max(16),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        self.queue().submit([encoder.finish()]);
        let _ = self.device().poll(wgpu::PollType::wait_indefinitely());
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device().poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map error");
        let mapped = slice.get_mapped_range().unwrap();
        mapped[..bytes as usize].to_vec()
    }

    fn collect_readbacks(&mut self) {
        for (step, bytes) in self.buffers.bodies_readback.poll(self.gpu.device()) {
            self.consume_bodies(step, &bytes);
        }
        for (step, bytes) in self.buffers.queries_readback.poll(self.gpu.device()) {
            self.consume_queries(step, &bytes);
        }
        for (step, bytes) in self.buffers.constraints_readback.poll(self.gpu.device()) {
            self.consume_constraints(step, &bytes);
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
                previous_position: record.prev_position,
                orientation: record.orientation,
                velocity: record.velocity,
                angular_velocity: record.angular_velocity,
                inverse_mass: record.inverse_mass,
                com: record.com,
                sleeping: record.flags & BODY_SLEEPING != 0,
                step,
            });
        }
    }

    fn consume_constraints(&mut self, _step: u64, bytes: &[u8]) {
        let records: &[ConstraintRecord] = bytemuck::cast_slice(bytes);
        let broken = records
            .iter()
            .enumerate()
            .filter(|(_, record)| record.kind == CONSTRAINT_INVALID)
            .map(|(slot, _)| slot)
            .collect::<Vec<_>>();
        for slot in broken.into_iter().rev() {
            if slot >= self.constraint_alive.len() {
                continue;
            }
            let handle = self.constraint_alive[slot];
            let record = self.constraint_records[slot];
            if record.kind != CONSTRAINT_INVALID {
                self.broken_constraints.push(handle);
                self.remove_constraint(handle);
            }
        }
    }

    pub fn drain_constraint_breaks(&mut self) -> Vec<ConstraintHandle> {
        std::mem::take(&mut self.broken_constraints)
    }

    fn island_rounds(&self) -> u32 {
        (self.capacity as u32).ilog2() + 1
    }
}
