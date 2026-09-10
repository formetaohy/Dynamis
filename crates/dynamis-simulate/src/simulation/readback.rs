use super::{ContactManifold, ContactPoint, Simulation};
use crate::buffers::EVENT_SLOTS;
use dynamis_layout::{
    BodyStateRecord, COUNTER_CONTACTS, COUNTER_EVENTS, ConstraintRuntimeRecord, ContactEventRecord,
    ContactRecord, Counters,
};
use dynamis_model::{BodyHandle, BodyState, ConstraintHandle};
use std::mem::size_of;

impl Simulation {
    pub fn poll(&mut self) {
        self.gpu.assert_alive();
        self.collect_readbacks();
    }

    pub fn wait(&mut self) {
        self.synchronize_states();
    }

    pub fn synchronize_states(&mut self) {
        self.gpu.assert_alive();
        self.gpu
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device lost while awaiting readback");
        self.gpu.assert_alive();
        self.collect_readbacks();
        if !self.states_synchronized {
            self.refresh_body_states();
            self.states_synchronized = true;
        }
    }

    fn refresh_body_states(&mut self) {
        let bytes = u64::from(self.device_body_count) * size_of::<BodyStateRecord>() as u64;
        if bytes == 0 {
            return;
        }
        let buffer = self.buffers.body_states.buffer().clone();
        let records = self.read_range(&buffer, bytes);
        let step = self.step_index.saturating_sub(1);
        for record in crate::records::records::<BodyStateRecord>(&records) {
            self.accept_body(step, record);
        }
    }

    /// What the device measured on the step whose pack last arrived.
    pub fn measured(&self) -> &Counters {
        &self.observed
    }

    pub fn contact_manifolds(&mut self) -> Vec<ContactManifold> {
        self.wait();
        let count = self.observed[COUNTER_CONTACTS] as usize;
        if count == 0 {
            return Vec::new();
        }
        let bytes = (count * size_of::<ContactRecord>()) as u64;
        let buffer = self.buffers.contacts.buffer().clone();
        let records = self.read_range(&buffer, bytes);
        crate::records::records::<ContactRecord>(&records)
            .into_iter()
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

    /// Reads `bytes` of a device buffer with a synchronous round trip. For inspection
    /// only; the step path never uses it. The staging buffer is reused across calls,
    /// so a per-frame state sync stops allocating on the hot path.
    pub(crate) fn read_range(&mut self, buffer: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        let device = self.gpu.device();
        let wide = bytes.max(16);
        if self.sync_staging.as_ref().is_none_or(|(staging, size)| {
            *size < wide || !staging.usage().contains(wgpu::BufferUsages::MAP_READ)
        }) {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dynamis sync readback"),
                size: wide,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.sync_staging = Some((staging, wide));
        }
        let staging = &self.sync_staging.as_ref().expect("staging just ensured").0;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, bytes);
        self.gpu.queue().submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().expect("map").expect("map error");
        let mapped = slice.get_mapped_range().unwrap();
        let data = mapped[..bytes as usize].to_vec();
        drop(mapped);
        staging.unmap();
        data
    }

    pub(crate) fn collect_readbacks(&mut self) {
        for (step, bytes) in self.buffers.readback.poll(self.gpu.device()) {
            self.consume_pack(step, &bytes);
        }
        for (step, bytes) in self.buffers.events_readback.poll(self.gpu.device()) {
            self.consume_events(step, &bytes);
        }
        for (batch, bytes) in self.buffers.queries_readback.poll(self.gpu.device()) {
            self.query_pool.collect(batch, &bytes);
        }
        #[cfg(feature = "profile")]
        for (_step, timings) in self.pipeline.poll_timings() {
            self.pass_timings = timings;
        }
    }

    pub(crate) fn drain_readbacks(&mut self) {
        for (step, bytes) in self.buffers.readback.drain(self.gpu.device()) {
            self.consume_pack(step, &bytes);
        }
        for (step, bytes) in self.buffers.events_readback.drain(self.gpu.device()) {
            self.consume_events(step, &bytes);
        }
        let device = self.gpu.device();
        for (batch, bytes) in self.buffers.queries_readback.drain(device) {
            self.query_pool.collect(batch, &bytes);
        }
    }

    /// `consume_pack` marks a step's events; a later encoder (or a forced readback)
    /// copies them home by the count the step actually produced.
    fn note_events_due(&mut self, step: u64) {
        let count = self.observed[COUNTER_EVENTS];
        if count > 0 {
            self.events_due.push_back((step, count));
        }
    }

    /// Copies every marked event segment into the readback staging, at the head of
    /// the encoder that will not overwrite them before the copy runs.
    pub(crate) fn copy_events(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
    ) {
        while let Some((step, count)) = self.events_due.pop_front() {
            let segment = self.buffers.events.size() / EVENT_SLOTS as u64;
            let offset = (step % EVENT_SLOTS as u64) * segment;
            let bytes = count as u64 * size_of::<ContactEventRecord>() as u64;
            let displaced = self.buffers.events_readback.enqueue(
                device,
                encoder,
                self.buffers.events.buffer(),
                offset,
                bytes,
                step,
            );
            if let Some((step, bytes)) = displaced {
                self.consume_events(step, &bytes);
            }
        }
    }

    /// Blocks until every event a landed pack announced is home, then consumes them.
    pub(crate) fn sync_events(&mut self) {
        if self.events_due.is_empty() {
            return;
        }
        let device = self.gpu.device().clone();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dynamis event readback"),
        });
        self.copy_events(&mut encoder, &device);
        self.gpu.queue().submit([encoder.finish()]);
        self.buffers.events_readback.arm();
        for (step, bytes) in self.buffers.events_readback.drain(&device) {
            self.consume_events(step, &bytes);
        }
    }

    /// Records what the device measured; the capacity controller widens or narrows
    /// the streams from this at the next step boundary, so a spill only ever degrades
    /// the step that produced it.
    pub(crate) fn accept_measured(&mut self, _step: u64, measured: &Counters) {
        self.observed = *measured;
        self.note_events_due(_step);
    }

    pub(crate) fn accept_body(&mut self, step: u64, record: BodyStateRecord) {
        let id = record.body_id as usize;
        assert!(
            id < self.slots,
            "GPU readback returned an out-of-range body id"
        );
        if record.generation != self.generations[id] || self.index_of[id] == u32::MAX {
            return;
        }
        self.states[id] = Some(BodyState {
            position: record.position,
            prev_position: record.prev_position,
            orientation: record.orientation,
            velocity: record.velocity,
            angular_velocity: record.angular_velocity,
            inverse_mass: self.descriptors[id].inverse_mass,
            com: self.descriptors[id].com,
            sleeping: record.sleeping != 0,
            step,
        });
    }

    pub(crate) fn accept_constraint_break(&mut self, _step: u64, record: ConstraintRuntimeRecord) {
        if record.broken == 0 {
            return;
        }
        let id = record.constraint_id as usize;
        if id >= self.constraint_generations.len() {
            return;
        }
        if self.constraint_generations[id] != record.generation
            || self.constraint_index_of[id] == u32::MAX
        {
            return;
        }
        let handle = ConstraintHandle {
            id: id as u32,
            generation: record.generation,
        };
        self.broken_constraints.push(handle);
        self.remove_constraint(handle);
    }

    pub fn drain_constraint_breaks(&mut self) -> Vec<ConstraintHandle> {
        std::mem::take(&mut self.broken_constraints)
    }

    pub(crate) fn island_rounds(&self) -> u32 {
        self.reservation.bodies.max(2).ilog2() + 1
    }
}
