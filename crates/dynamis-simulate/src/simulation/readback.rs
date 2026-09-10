use super::Simulation;
use crate::buffers::EVENT_SLOTS;
use dynamis_layout::{
    BodyStateRecord, COUNTER_CONSTRAINTS, COUNTER_CONTACTS, COUNTER_COUNT, COUNTER_EVENTS,
    COUNTER_STRIDE, ConstraintRuntimeRecord, ContactEventRecord, ContactRecord, Counters,
};
use dynamis_model::{BodyHandle, BodyState, ConstraintHandle};
use std::mem::size_of;

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

const COUNTER_BYTES: u64 = COUNTER_STRIDE * COUNTER_COUNT as u64;

fn pack_bytes(constraints: u32) -> u64 {
    COUNTER_BYTES + constraints as u64 * size_of::<ConstraintRuntimeRecord>() as u64
}

fn measured_counters(bytes: &[u8]) -> Counters {
    let stride = COUNTER_STRIDE as usize;
    let mut counters: Counters = [0; COUNTER_COUNT];
    for (slot, value) in counters.iter_mut().enumerate() {
        let at = slot * stride;
        *value = u32::from_le_bytes(bytes[at..at + 4].try_into().expect("counter slot"));
    }
    counters
}

impl Simulation {
    /// One pack per step: the counter vector, then the constraint runtime rows. Contact
    /// events travel in their own readback, sized by what the step actually produced, so
    /// an idle event stream never bills its whole forecast.
    pub(crate) fn pack_step(&self, encoder: &mut wgpu::CommandEncoder) -> u64 {
        let staging = self.device.buffers.readback.pack.buffer();
        encoder.copy_buffer_to_buffer(
            self.device.buffers.counters.buffer(),
            0,
            staging,
            0,
            COUNTER_BYTES,
        );
        let constraints = self.constraints.alive.len() as u32;
        if constraints > 0 {
            encoder.copy_buffer_to_buffer(
                self.device.buffers.constraints.runtime.buffer(),
                0,
                staging,
                COUNTER_BYTES,
                constraints as u64 * size_of::<ConstraintRuntimeRecord>() as u64,
            );
        }
        pack_bytes(constraints)
    }

    pub(crate) fn consume_pack(&mut self, step: u64, bytes: &[u8]) {
        let measured = measured_counters(bytes);
        self.accept_measured(step, &measured);
        let rows =
            measured[COUNTER_CONSTRAINTS] as u64 * size_of::<ConstraintRuntimeRecord>() as u64;
        let records =
            crate::records::decode::<ConstraintRuntimeRecord>(range(bytes, COUNTER_BYTES, rows));
        for record in records {
            self.accept_constraint_break(record);
        }
    }

    pub fn poll(&mut self) {
        self.device.gpu.assert_alive();
        self.collect_readbacks();
    }

    pub fn wait(&mut self) {
        self.synchronize_states();
    }

    pub fn synchronize_states(&mut self) {
        self.device.gpu.assert_alive();
        self.device
            .gpu
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device lost while awaiting readback");
        self.device.gpu.assert_alive();
        self.collect_readbacks();
        if !self.bodies.states_ready {
            self.refresh_body_states();
            self.bodies.states_ready = true;
        }
    }

    fn refresh_body_states(&mut self) {
        let bytes = u64::from(self.bodies.device_count) * size_of::<BodyStateRecord>() as u64;
        if bytes == 0 {
            return;
        }
        let buffer = self.device.buffers.bodies.states.buffer().clone();
        let records = self.read_range(&buffer, bytes);
        let step = self.clock.step.saturating_sub(1);
        for record in crate::records::decode::<BodyStateRecord>(&records) {
            self.accept_body(step, record);
        }
    }

    /// What the device measured on the step whose pack last arrived.
    pub fn measured(&self) -> &Counters {
        &self.device.measured
    }

    pub fn contact_manifolds(&mut self) -> Vec<ContactManifold> {
        self.wait();
        let count = self.device.measured[COUNTER_CONTACTS] as usize;
        if count == 0 {
            return Vec::new();
        }
        let bytes = (count * size_of::<ContactRecord>()) as u64;
        let buffer = self.device.buffers.contacts.manifolds.buffer().clone();
        let records = self.read_range(&buffer, bytes);
        crate::records::decode::<ContactRecord>(&records)
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
                step: self.clock.step.saturating_sub(1),
            })
            .collect()
    }

    /// Reads `bytes` of a device buffer with a synchronous round trip. For inspection
    /// only; the step path never uses it. The staging buffer is reused across calls,
    /// so a per-frame state sync stops allocating on the hot path.
    pub(crate) fn read_range(&mut self, buffer: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        let device = self.device.gpu.device();
        let wide = bytes.max(16);
        if self
            .device
            .sync_staging
            .as_ref()
            .is_none_or(|(_, size)| *size < wide)
        {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dynamis sync readback"),
                size: wide,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.device.sync_staging = Some((staging, wide));
        }
        let staging = &self
            .device
            .sync_staging
            .as_ref()
            .expect("staging just ensured")
            .0;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, bytes);
        self.device.gpu.queue().submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device lost while reading back");
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device lost while reading back");
        receiver
            .recv()
            .expect("readback mapping sender dropped")
            .expect("readback mapping failed");
        let mapped = slice.get_mapped_range().expect("mapped range unavailable");
        let data = mapped[..bytes as usize].to_vec();
        drop(mapped);
        staging.unmap();
        data
    }

    pub(crate) fn collect_readbacks(&mut self) {
        for (step, bytes) in self
            .device
            .buffers
            .readback
            .step
            .poll(self.device.gpu.device())
        {
            self.consume_pack(step, &bytes);
        }
        for (_, bytes) in self
            .device
            .buffers
            .readback
            .events
            .poll(self.device.gpu.device())
        {
            self.consume_events(&bytes);
        }
        for (batch, bytes) in self
            .device
            .buffers
            .readback
            .queries
            .poll(self.device.gpu.device())
        {
            self.queries.pool.collect(batch, &bytes);
        }
        #[cfg(feature = "profile")]
        for (_step, timings) in self.device.pipeline.poll_timings() {
            self.device.pass_timings = timings;
        }
    }

    pub(crate) fn drain_readbacks(&mut self) {
        for (step, bytes) in self
            .device
            .buffers
            .readback
            .step
            .drain(self.device.gpu.device())
        {
            self.consume_pack(step, &bytes);
        }
        for (_, bytes) in self
            .device
            .buffers
            .readback
            .events
            .drain(self.device.gpu.device())
        {
            self.consume_events(&bytes);
        }
        let device = self.device.gpu.device();
        for (batch, bytes) in self.device.buffers.readback.queries.drain(device) {
            self.queries.pool.collect(batch, &bytes);
        }
    }

    /// `consume_pack` marks a step's events; a later encoder (or a forced readback)
    /// copies them home by the count the step actually produced.
    fn note_events_due(&mut self, step: u64) {
        let count = self.device.measured[COUNTER_EVENTS];
        if count > 0 {
            self.events.due.push_back((step, count));
        }
    }

    /// Copies every marked event segment into the readback staging, at the head of
    /// the encoder that will not overwrite them before the copy runs.
    pub(crate) fn copy_events(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
    ) {
        while let Some((step, count)) = self.events.due.pop_front() {
            let segment = self.device.buffers.events.size() / EVENT_SLOTS as u64;
            let offset = (step % EVENT_SLOTS as u64) * segment;
            let bytes = count as u64 * size_of::<ContactEventRecord>() as u64;
            let displaced = self.device.buffers.readback.events.enqueue(
                device,
                encoder,
                self.device.buffers.events.buffer(),
                offset,
                bytes,
                step,
            );
            if let Some((_, bytes)) = displaced {
                self.consume_events(&bytes);
            }
        }
    }

    /// Blocks until every event a landed pack announced is home, then consumes them.
    pub(crate) fn sync_events(&mut self) {
        if self.events.due.is_empty() {
            return;
        }
        let device = self.device.gpu.device().clone();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dynamis event readback"),
        });
        self.copy_events(&mut encoder, &device);
        self.device.gpu.queue().submit([encoder.finish()]);
        self.device.buffers.readback.events.arm();
        for (_, bytes) in self.device.buffers.readback.events.drain(&device) {
            self.consume_events(&bytes);
        }
    }

    /// Records what the device measured; the capacity controller widens or narrows
    /// the streams from this at the next step boundary, so a spill only ever degrades
    /// the step that produced it.
    pub(crate) fn accept_measured(&mut self, step: u64, measured: &Counters) {
        self.device.measured = *measured;
        self.note_events_due(step);
    }

    pub(crate) fn accept_body(&mut self, step: u64, record: BodyStateRecord) {
        let id = record.body_id as usize;
        assert!(
            id < self.slots,
            "GPU readback returned an out-of-range body id"
        );
        if record.generation != self.bodies.generations[id] || self.bodies.index_of[id] == u32::MAX
        {
            return;
        }
        self.bodies.states[id] = Some(BodyState {
            position: record.position,
            prev_position: record.prev_position,
            orientation: record.orientation,
            velocity: record.velocity,
            angular_velocity: record.angular_velocity,
            inverse_mass: self.bodies.descriptors[id].inverse_mass,
            com: self.bodies.descriptors[id].com,
            sleeping: record.sleeping != 0,
            step,
        });
    }

    pub(crate) fn accept_constraint_break(&mut self, record: ConstraintRuntimeRecord) {
        if record.broken == 0 {
            return;
        }
        let id = record.constraint_id as usize;
        if id >= self.constraints.generations.len() {
            return;
        }
        if self.constraints.generations[id] != record.generation
            || self.constraints.index_of[id] == u32::MAX
        {
            return;
        }
        let handle = ConstraintHandle {
            id: id as u32,
            generation: record.generation,
        };
        self.constraints.broken.push(handle);
        self.remove_constraint(handle);
    }

    pub fn drain_constraint_breaks(&mut self) -> Vec<ConstraintHandle> {
        std::mem::take(&mut self.constraints.broken)
    }

    pub(crate) fn island_rounds(&self) -> u32 {
        self.device.reservation.bodies.max(2).ilog2() + 1
    }
}

fn range(bytes: &[u8], at: u64, len: u64) -> &[u8] {
    let at = at as usize;
    &bytes[at..at + len as usize]
}
