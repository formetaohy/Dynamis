use super::World;
use dynamis_layout::COUNTER_RESTING;
use dynamis_layout::{
    COUNTER_CONSTRAINTS, COUNTER_CONTACTS, COUNTER_COUNT, COUNTER_STRIDE, ConstraintRuntimeRecord,
    ContactRecord, Counters,
};
use dynamis_model::{BodyHandle, ConstraintHandle};
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
const CONTACT_BYTES: u64 = size_of::<ContactRecord>() as u64;

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

    pub fn contact_manifolds(&mut self) -> Vec<ContactManifold> {
        self.wait();
        let step = self.clock.step.saturating_sub(1);
        let active = self.backend.measured[COUNTER_CONTACTS] as usize;
        let capacity =
            (self.backend.streams.rigid.resting_contacts.size() / CONTACT_BYTES) as usize;
        let resting = (self.backend.measured[COUNTER_RESTING] as usize).min(capacity);
        if active == 0 && resting == 0 {
            return Vec::new();
        }
        let active_buffer = self.backend.streams.rigid.contacts.buffer().clone();
        let resting_buffer = self.backend.streams.rigid.resting_contacts.buffer().clone();
        let resting_live = self.backend.streams.rigid.resting_live.buffer().clone();
        let mut regions = Vec::with_capacity(3);
        if active > 0 {
            regions.push((&active_buffer, 0, active as u64 * CONTACT_BYTES));
        }
        if resting > 0 {
            regions.push((&resting_live, 0, resting as u64 * 4));
            regions.push((&resting_buffer, 0, resting as u64 * CONTACT_BYTES));
        }
        let bytes = self.read_regions("world contact readback", &regions);
        let mut manifolds = Vec::with_capacity(active + resting);
        let mut seen = HashSet::new();
        let active_bytes = active * size_of::<ContactRecord>();
        for record in dynamis_layout::decode::<ContactRecord>(&bytes[..active_bytes]) {
            if seen.insert((record.a, record.b)) {
                manifolds.push(manifold_of(&record, step));
            }
        }
        if resting > 0 {
            let live = &bytes[active_bytes..active_bytes + resting * 4];
            let resting_bytes = &bytes[active_bytes + resting * 4..];
            for (index, record) in dynamis_layout::decode::<ContactRecord>(resting_bytes)
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
        }
        manifolds
    }

    pub(crate) fn read_regions(
        &mut self,
        label: &str,
        regions: &[(&wgpu::Buffer, u64, u64)],
    ) -> Vec<u8> {
        let bytes: u64 = regions.iter().map(|region| region.2).sum();
        assert!(
            bytes > 0 && bytes.is_multiple_of(4),
            "an inspection read must cover a positive word aligned length"
        );
        let device = self.backend.gpu.device().clone();
        let mut readback = match self.backend.inspect.take() {
            Some(readback) if readback.size() >= bytes => readback,
            _ => dynamis_gpu::Readback::new(&device, "world inspection readback", bytes, 1),
        };
        let mut encoder = dynamis_gpu::SubmissionEncoder::new(&device, label);
        assert!(
            readback.enqueue_regions(&mut encoder, regions, 0).is_none(),
            "an inspection read requires an idle readback"
        );
        self.submit(encoder);
        let entry = readback
            .drain()
            .pop()
            .expect("an inspection read retires exactly once");
        self.backend.inspect = Some(readback);
        entry.1
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
        for (step, bytes) in self.backend.streams.readback.states.collect() {
            self.consume_states(step, &bytes);
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
        for (step, bytes) in self.backend.streams.readback.states.drain() {
            self.consume_states(step, &bytes);
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

    pub fn measured(&self) -> &Counters {
        &self.backend.measured
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
