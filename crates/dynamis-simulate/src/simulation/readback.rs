use super::{ContactManifold, ContactPoint, Simulation};
use dynamis_layout::{
    BodyStateRecord, COUNTER_CONTACTS, COUNTER_SPILLOVER_ENTRIES, COUNTER_SPILLOVER_EVENTS,
    COUNTER_SPILLOVER_PAIRS, ConstraintRuntimeRecord, ContactRecord, Counters,
};
use dynamis_model::{BodyHandle, BodyState, ConstraintHandle};
use std::mem::size_of;

impl Simulation {
    pub fn poll(&mut self) {
        self.gpu.assert_alive();
        self.collect_readbacks();
    }

    pub fn wait(&mut self) {
        self.gpu.assert_alive();
        self.gpu
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device lost while awaiting readback");
        self.gpu.assert_alive();
        self.collect_readbacks();
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
        let records = self.read_range(self.buffers.contacts.buffer(), bytes);
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
    /// only; the step path never uses it.
    pub(crate) fn read_range(&self, buffer: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        let staging = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("dynamis sync readback"),
            size: bytes.max(16),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        self.queue().submit([encoder.finish()]);
        let _ = self.device().poll(wgpu::PollType::wait_indefinitely());
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = self.device().poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().expect("map").expect("map error");
        let mapped = slice.get_mapped_range().unwrap();
        mapped[..bytes as usize].to_vec()
    }

    pub(crate) fn collect_readbacks(&mut self) {
        for (step, bytes) in self.buffers.readback.poll(self.gpu.device()) {
            self.consume_pack(step, &bytes);
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
        let device = self.gpu.device();
        for (batch, bytes) in self.buffers.queries_readback.drain(device) {
            self.query_pool.collect(batch, &bytes);
        }
    }

    /// Adopts a measured vector and fails the moment a stream ran out of room, so a
    /// shortfall surfaces at the step that hit it rather than as missing contacts.
    pub(crate) fn accept_measured(&mut self, _step: u64, measured: &Counters) {
        assert_eq!(
            measured[COUNTER_SPILLOVER_PAIRS],
            0,
            "the pair stream held {} lanes, this step needed {}",
            self.reservation.pairs,
            self.reservation
                .pairs
                .saturating_add(measured[COUNTER_SPILLOVER_PAIRS])
        );
        assert_eq!(
            measured[COUNTER_SPILLOVER_ENTRIES],
            0,
            "the entry stream held {} cells, this step needed {}",
            self.reservation.entries,
            self.reservation
                .entries
                .saturating_add(measured[COUNTER_SPILLOVER_ENTRIES])
        );
        assert_eq!(
            measured[COUNTER_SPILLOVER_EVENTS],
            0,
            "the event stream held {} lanes, this step needed {}",
            self.reservation.events,
            self.reservation
                .events
                .saturating_add(measured[COUNTER_SPILLOVER_EVENTS])
        );
        self.observed = *measured;
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
