use super::World;
use dynamis_layout::COUNTER_RESTING;
use dynamis_layout::{
    BodyStateRecord, COUNTER_CONSTRAINTS, COUNTER_CONTACTS, COUNTER_COUNT, COUNTER_STRIDE,
    ConstraintRuntimeRecord, ContactRecord, Counters,
};
use dynamis_model::{BodyHandle, BodyState, ConstraintHandle};
use std::collections::HashSet;
use std::mem::size_of;

pub struct ContactPoint {
    pub position: [f32; 3],
    pub depth: f32,
    pub normal_impulse: f32,
    pub tangent_impulse: f32,
    pub feature: u32,
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

impl World {
    pub(crate) fn pack_step(&self, encoder: &mut wgpu::CommandEncoder) -> u64 {
        let staging = self.backend.streams.readback.pack.buffer();
        encoder.copy_buffer_to_buffer(
            self.backend.streams.scene.counters.buffer(),
            0,
            staging,
            0,
            COUNTER_BYTES,
        );
        let constraints = self.constraints.alive.len() as u32;
        if constraints > 0 {
            encoder.copy_buffer_to_buffer(
                self.backend.streams.scene.constraint_runtime.buffer(),
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
            dynamis_layout::decode::<ConstraintRuntimeRecord>(range(bytes, COUNTER_BYTES, rows));
        for record in records {
            self.accept_constraint_break(record);
        }
    }

    pub fn poll(&mut self) {
        self.backend.gpu.assert_alive();
        self.collect_readbacks();
    }

    pub fn wait(&mut self) {
        self.synchronize_states();
    }

    pub fn synchronize_states(&mut self) {
        self.backend.gpu.assert_alive();
        self.drain_readbacks();
        self.backend.gpu.assert_alive();
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
        let buffer = self.backend.streams.scene.body_states.buffer().clone();
        let records = self.read_range(&buffer, bytes);
        let step = self.clock.step.saturating_sub(1);
        for record in dynamis_layout::decode::<BodyStateRecord>(&records) {
            self.accept_body(step, record);
        }
    }

    pub fn measured(&self) -> &Counters {
        &self.backend.measured
    }

    pub fn contact_manifolds(&mut self) -> Vec<ContactManifold> {
        self.wait();
        let step = self.clock.step.saturating_sub(1);
        let active = self.backend.measured[COUNTER_CONTACTS] as usize;
        let capacity = (self.backend.streams.rigid.resting_contacts.size()
            / size_of::<ContactRecord>() as u64) as usize;
        let resting = (self.backend.measured[COUNTER_RESTING] as usize).min(capacity);
        let mut seen = HashSet::new();
        let mut manifolds = Vec::with_capacity(active + resting);
        let active_buffer = self.backend.streams.rigid.contacts.buffer().clone();
        for record in self.read_manifolds(&active_buffer, active) {
            if seen.insert((record.a, record.b)) {
                manifolds.push(manifold_of(&record, step));
            }
        }
        let resting_buffer = self.backend.streams.rigid.resting_contacts.buffer().clone();
        let resting_live = self.backend.streams.rigid.resting_live.buffer().clone();
        let live = self.read_range(&resting_live, resting as u64 * 4);
        for (index, record) in self
            .read_manifolds(&resting_buffer, resting)
            .into_iter()
            .enumerate()
        {
            if live[index * 4..index * 4 + 4] == [0, 0, 0, 0] {
                continue;
            }
            if seen.insert((record.a, record.b)) {
                manifolds.push(manifold_of(&record, step));
            }
        }
        manifolds
    }

    fn read_manifolds(&mut self, buffer: &wgpu::Buffer, count: usize) -> Vec<ContactRecord> {
        if count == 0 {
            return Vec::new();
        }
        let bytes = (count * size_of::<ContactRecord>()) as u64;
        let records = self.read_range(buffer, bytes);
        dynamis_layout::decode::<ContactRecord>(&records)
    }

    pub(crate) fn read_range(&mut self, buffer: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        if bytes == 0 {
            return Vec::new();
        }
        if self
            .backend
            .state_readback
            .as_ref()
            .is_none_or(|readback| readback.size() < bytes)
        {
            self.backend.state_readback = Some(dynamis_gpu::BufferReadback::new(
                self.backend.gpu.device(),
                "dynamis state readback",
                bytes,
            ));
        }
        self.backend
            .state_readback
            .as_mut()
            .expect("state readback just allocated")
            .read(self.backend.gpu.queue(), buffer, 0, bytes)
    }

    pub(crate) fn collect_readbacks(&mut self) {
        self.backend.gpu.poll();
        for (step, bytes) in self.backend.streams.readback.step.collect() {
            self.consume_pack(step, &bytes);
        }
        for (_, bytes) in self.backend.streams.readback.events.collect() {
            self.consume_events(&bytes);
        }
        for (batch, bytes) in self.backend.streams.readback.queries.collect() {
            self.queries.pool.collect(batch, &bytes);
        }
        #[cfg(feature = "profile")]
        for (_step, timings) in self.backend.pipeline.collect_timings() {
            self.backend.pass_timings = timings;
        }
    }

    pub(crate) fn drain_readbacks(&mut self) {
        for (step, bytes) in self.backend.streams.readback.step.drain() {
            self.consume_pack(step, &bytes);
        }
        for (_, bytes) in self.backend.streams.readback.events.drain() {
            self.consume_events(&bytes);
        }
        for (batch, bytes) in self.backend.streams.readback.queries.drain() {
            self.queries.pool.collect(batch, &bytes);
        }
        #[cfg(feature = "profile")]
        for (_step, timings) in self.backend.pipeline.collect_timings() {
            self.backend.pass_timings = timings;
        }
    }

    pub(crate) fn accept_measured(&mut self, step: u64, measured: &Counters) {
        self.backend.measured = *measured;
        self.backend.measured_step = Some(step);
        self.note_events_due(step);
    }

    pub(crate) fn accept_body(&mut self, step: u64, record: BodyStateRecord) {
        let id = record.body_id as usize;
        assert!(
            id < self.bodies.ids.len(),
            "GPU readback returned an out-of-range body id"
        );
        if record.generation != self.bodies.ids.generation(record.body_id)
            || self.bodies.index_of[id] == u32::MAX
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
        if id >= self.constraints.ids.len() {
            return;
        }
        if self.constraints.ids.generation(record.constraint_id) != record.generation
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
}

fn manifold_of(record: &ContactRecord, step: u64) -> ContactManifold {
    ContactManifold {
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
                tangent_impulse: (point.accumulated_tangent_1 * point.accumulated_tangent_1
                    + point.accumulated_tangent_2 * point.accumulated_tangent_2)
                    .sqrt(),
                feature: point.feature,
            })
            .collect(),
        step,
    }
}

fn range(bytes: &[u8], at: u64, len: u64) -> &[u8] {
    let at = at as usize;
    &bytes[at..at + len as usize]
}
