use super::{ContactManifold, ContactPoint, Simulation};
use dynamis_layout::{
    BODY_SLEEPING, CONSTRAINT_INVALID, ConstraintRecord, ContactRecord, RigidBodyRecord,
};
use dynamis_model::{BodyHandle, BodyState};

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

    pub fn contact_manifolds(&mut self) -> Vec<ContactManifold> {
        self.wait();
        let count_bytes = std::mem::size_of::<dynamis_layout::Counter>();
        let args: Vec<dynamis_layout::Counter> = bytemuck::cast_slice(
            &self.read_gpu(self.buffers.contact_count.buffer(), count_bytes as u64),
        )
        .to_vec();
        let count = (args[0].count as usize).min(self.buffers.contact_capacity() as usize);
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

    pub(crate) fn collect_readbacks(&mut self) {
        for (step, bytes) in self.buffers.bodies_readback.poll(self.gpu.device()) {
            self.consume_bodies(step, &bytes);
        }
        for (step, bytes) in self.buffers.queries_readback.poll(self.gpu.device()) {
            self.consume_queries(step, &bytes);
        }
        for (_step, bytes) in self.buffers.events_readback.poll(self.gpu.device()) {
            self.consume_events(&bytes);
        }
        for (step, bytes) in self.buffers.constraints_readback.poll(self.gpu.device()) {
            self.consume_constraints(step, &bytes);
        }
        #[cfg(feature = "profile")]
        for (_step, timings) in self.pipeline.poll_timings(self.gpu.device()) {
            self.pass_timings = timings;
        }
    }

    pub(crate) fn consume_bodies(&mut self, step: u64, bytes: &[u8]) {
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
                prev_position: record.prev_position,
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

    pub(crate) fn consume_constraints(&mut self, _step: u64, bytes: &[u8]) {
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

    pub fn drain_constraint_breaks(&mut self) -> Vec<dynamis_model::ConstraintHandle> {
        std::mem::take(&mut self.broken_constraints)
    }
}
