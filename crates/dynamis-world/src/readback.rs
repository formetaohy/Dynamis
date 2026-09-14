use super::World;
use dynamis_abi::COUNTER_RESTING;
use dynamis_abi::{
    COUNTER_CONTACTS, COUNTER_DEVICE_COUNT, COUNTER_STRIDE, ConstraintReactionRecord,
    ConstraintRuntimeRecord, ContactRecord, Counters, DeclaredCounters, FEATURE_KIND_MASK,
    FEATURE_TRIANGLE, FEATURE_TRIANGLE_MASK, JointStateRecord, NO_SURFACE, SHAPE_HEIGHTFIELD,
    SHAPE_MESH,
};
use dynamis_model::{BodyHandle, ConstraintHandle, JointState, SurfaceDesc};
use std::collections::HashSet;
use std::mem::size_of;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactPoint {
    pub position: [f32; 3],
    pub depth: f32,
    pub normal_impulse: f32,
    pub tangent_impulse: f32,
    pub feature: u32,
    pub triangle: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContactManifold {
    pub first: BodyHandle,
    pub second: BodyHandle,
    pub sensor: bool,
    pub normal: [f32; 3],
    pub material: SurfaceDesc,
    pub surface: Option<SurfaceDesc>,
    pub points: Vec<ContactPoint>,
    pub step: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstraintForce {
    pub constraint: ConstraintHandle,
    pub first: BodyHandle,
    pub second: BodyHandle,
    pub force_on_second: [f32; 3],
    pub torque_on_second: [f32; 3],
}

const COUNTER_BYTES: u64 = COUNTER_STRIDE * COUNTER_DEVICE_COUNT as u64;
const CONTACT_BYTES: u64 = size_of::<ContactRecord>() as u64;
const DECLARED_DEPTH: usize = dynamis_gpu::FACT_LAG + 2;

fn measured_counters(bytes: &[u8], counters: &mut Counters) {
    let stride = COUNTER_STRIDE as usize;
    for (slot, value) in counters.iter_mut().enumerate().take(COUNTER_DEVICE_COUNT) {
        let at = slot * stride;
        *value = u32::from_le_bytes(bytes[at..at + 4].try_into().expect("counter slot"));
    }
}

impl World {
    pub(crate) fn pack_step(&self, encoder: &mut wgpu::CommandEncoder) -> u64 {
        encoder.copy_buffer_to_buffer(
            self.backend.streams.state.counters.buffer(),
            0,
            self.backend.readback.pack.buffer(),
            0,
            COUNTER_BYTES,
        );
        COUNTER_BYTES
    }

    pub(crate) fn declare_step(&mut self, step: u64) {
        let declared = DeclaredCounters {
            bodies: self.bodies.alive.len() as u32,
            colliders: self.colliders.live(),
            constraints: self.constraints.alive.len() as u32,
            body_edits: self.bodies.last_edits,
            body_moves: self.bodies.last_moves,
            constraint_commands: self.constraints.last_commands,
            constraint_moves: self.constraints.last_moves,
        };
        assert!(
            self.backend.declared.len() < DECLARED_DEPTH,
            "a step declaration outlived its counter readback"
        );
        self.backend.declared.push_back((step, declared));
    }

    pub(crate) fn consume_pack(&mut self, step: u64, bytes: &[u8]) {
        let (declared_step, declared) = self
            .backend
            .declared
            .pop_front()
            .expect("a counter readback retires a declared step");
        assert_eq!(
            declared_step, step,
            "counter readbacks must retire in declaration order"
        );
        declared.write_into(&mut self.backend.measured);
        measured_counters(bytes, &mut self.backend.measured);
        self.accept_measured(step);
    }

    pub fn inspect_constraint_forces(&mut self) -> Vec<ConstraintForce> {
        let count = self.constraints.alive.len();
        if count == 0 {
            return Vec::new();
        }
        self.wait();
        let runtime = &self.backend.streams.state.constraint_runtime;
        let bytes = count as u64 * runtime.stride();
        let buffer = runtime.buffer().clone();
        let read = self.read_regions("constraint force readback", &[(&buffer, 0, bytes)]);
        dynamis_abi::decode::<ConstraintRuntimeRecord>(&read)
            .into_iter()
            .enumerate()
            .map(|(row, runtime)| {
                let constraint = self.constraints.alive[row];
                let (first, second) = self.constraint_bodies(constraint);
                constraint_force_of(
                    constraint,
                    first,
                    second,
                    runtime.reaction,
                    self.clock.sub_dt,
                )
            })
            .collect()
    }

    pub fn inspect_constraint_force(&mut self, handle: ConstraintHandle) -> ConstraintForce {
        self.validate_constraint(handle);
        let row = self.constraints.index_of[handle.id as usize];
        self.wait();
        let runtime = &self.backend.streams.state.constraint_runtime;
        let stride = runtime.stride();
        let buffer = runtime.buffer().clone();
        let read = self.read_regions(
            "constraint force readback",
            &[(&buffer, u64::from(row) * stride, stride)],
        );
        let record = dynamis_abi::decode::<ConstraintRuntimeRecord>(&read)
            .first()
            .copied()
            .expect("a constraint force read covers exactly one record");
        let (first, second) = self.constraint_bodies(handle);
        constraint_force_of(handle, first, second, record.reaction, self.clock.sub_dt)
    }

    pub fn inspect_joint_states(&mut self) -> Vec<(ConstraintHandle, JointState)> {
        let count = self.constraints.alive.len();
        if count == 0 {
            return Vec::new();
        }
        self.wait();
        let states = &self.backend.streams.rigid.joint_states;
        let bytes = count as u64 * states.stride();
        let buffer = states.buffer().clone();
        let read = self.read_regions("joint state readback", &[(&buffer, 0, bytes)]);
        dynamis_abi::decode::<JointStateRecord>(&read)
            .into_iter()
            .enumerate()
            .map(|(row, state)| (self.constraints.alive[row], self.joint_state_at(row, state)))
            .collect()
    }

    pub fn inspect_joint_state(&mut self, handle: ConstraintHandle) -> JointState {
        self.validate_constraint(handle);
        let row = self.constraints.index_of[handle.id as usize];
        self.wait();
        let states = &self.backend.streams.rigid.joint_states;
        let stride = states.stride();
        let buffer = states.buffer().clone();
        let read = self.read_regions(
            "joint state readback",
            &[(&buffer, u64::from(row) * stride, stride)],
        );
        let state = dynamis_abi::decode::<JointStateRecord>(&read)
            .first()
            .copied()
            .expect("a joint state read covers exactly one record");
        self.joint_state_at(row as usize, state)
    }

    fn joint_state_at(&self, row: usize, state: JointStateRecord) -> JointState {
        joint_state_of(self.constraints.records[row].constraint_kind(), state)
    }

    pub fn inspect_contacts(&mut self) -> Vec<ContactManifold> {
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
        for record in dynamis_abi::decode::<ContactRecord>(&bytes[..active_bytes]) {
            if seen.insert((record.a, record.b)) {
                manifolds.push(manifold_of(&record, step, self.contact_surface(&record)));
            }
        }
        if resting > 0 {
            let live = &bytes[active_bytes..active_bytes + resting * 4];
            let resting_bytes = &bytes[active_bytes + resting * 4..];
            for (index, record) in dynamis_abi::decode::<ContactRecord>(resting_bytes)
                .into_iter()
                .enumerate()
            {
                if live[index * 4..index * 4 + 4] == [0, 0, 0, 0] {
                    continue;
                }
                if seen.insert((record.a, record.b)) {
                    manifolds.push(manifold_of(&record, step, self.contact_surface(&record)));
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
        self.collect_observations();
        for (step, bytes) in self.backend.readback.step.collect() {
            self.consume_pack(step, &bytes);
        }
        for (_, bytes) in self.backend.readback.events.collect() {
            self.consume_events(&bytes);
        }
        for (_, bytes) in self.backend.readback.breaks.collect() {
            self.consume_breaks(&bytes);
        }
        for (batch, bytes) in self.backend.readback.queries.collect() {
            self.collect_query_batch(batch, &bytes);
        }
        for (sequence, bytes) in self.backend.readback.collect_states() {
            self.consume_states(sequence, &bytes);
        }
        #[cfg(feature = "profile")]
        for timings in self.backend.passes.collect_timings() {
            self.backend.pass_timings = timings;
        }
    }

    pub(crate) fn drain_readbacks(&mut self) {
        self.drain_observations();
        for (step, bytes) in self.backend.readback.step.drain() {
            self.consume_pack(step, &bytes);
        }
        for (_, bytes) in self.backend.readback.events.drain() {
            self.consume_events(&bytes);
        }
        for (_, bytes) in self.backend.readback.breaks.drain() {
            self.consume_breaks(&bytes);
        }
        for (batch, bytes) in self.backend.readback.queries.drain() {
            self.collect_query_batch(batch, &bytes);
        }
        for (sequence, bytes) in self.backend.readback.drain_states() {
            self.consume_states(sequence, &bytes);
        }
        #[cfg(feature = "profile")]
        for timings in self.backend.passes.collect_timings() {
            self.backend.pass_timings = timings;
        }
    }

    pub(crate) fn accept_measured(&mut self, step: u64) {
        self.assert_no_device_faults();
        self.backend.measured_step = Some(step);
        self.note_events_due(step);
        self.note_breaks_due(step);
    }

    fn assert_no_device_faults(&self) {
        assert_eq!(
            self.backend.measured[dynamis_abi::COUNTER_ENTRY_FAULTS],
            0,
            "a collider or particle spans more grid cells than one entry budget holds"
        );
        assert_eq!(
            self.backend.measured[dynamis_abi::COUNTER_LIVE_FAULTS],
            0,
            "the live body set outgrew the planned body stream"
        );
    }

    pub fn measured(&self) -> &Counters {
        &self.backend.measured
    }

    pub(crate) fn contact_surface(&self, record: &ContactRecord) -> Option<SurfaceDesc> {
        if record.surface == NO_SURFACE {
            return None;
        }
        for slot in [record.a, record.b] {
            let collider = self.colliders.records()[slot as usize];
            if collider.kind == SHAPE_MESH || collider.kind == SHAPE_HEIGHTFIELD {
                return Some(
                    self.shapes
                        .pool
                        .source_surface(collider.source, record.surface),
                );
            }
        }
        panic!("a contact surface must belong to its scene geometry");
    }

    pub(crate) fn accept_constraint_break(&mut self, constraint_id: u32, generation: u32) {
        let id = constraint_id as usize;
        if id >= self.constraints.ids.len() {
            return;
        }
        if self.constraints.ids.generation(constraint_id) != generation
            || self.constraints.index_of[id] == u32::MAX
        {
            return;
        }
        let handle = ConstraintHandle {
            id: constraint_id,
            generation,
        };
        self.constraints.broken.push(handle);
        self.remove_constraint(handle);
    }
}

pub(crate) fn joint_state_of(
    kind: dynamis_model::ConstraintKind,
    state: JointStateRecord,
) -> JointState {
    assert_eq!(
        state.dof_count as usize,
        kind.dofs().len(),
        "the device and the host must agree on the {kind:?} dof layout"
    );
    JointState::new(kind, state.coordinates, state.rates, state.impulses)
}

pub(crate) fn constraint_force_of(
    constraint: ConstraintHandle,
    first: BodyHandle,
    second: BodyHandle,
    reaction: ConstraintReactionRecord,
    step_dt: f32,
) -> ConstraintForce {
    ConstraintForce {
        constraint,
        first,
        second,
        force_on_second: [
            reaction.linear_second[0] / step_dt,
            reaction.linear_second[1] / step_dt,
            reaction.linear_second[2] / step_dt,
        ],
        torque_on_second: [
            reaction.angular_second[0] / step_dt,
            reaction.angular_second[1] / step_dt,
            reaction.angular_second[2] / step_dt,
        ],
    }
}

fn feature_triangle(feature: u32) -> Option<u32> {
    ((feature & FEATURE_KIND_MASK) == FEATURE_TRIANGLE).then_some(feature & FEATURE_TRIANGLE_MASK)
}

fn manifold_of(record: &ContactRecord, step: u64, surface: Option<SurfaceDesc>) -> ContactManifold {
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
        material: SurfaceDesc {
            friction: record.friction,
            restitution: record.restitution,
            rolling_friction: record.rolling_friction,
            spin_friction: record.spin_friction,
        },
        surface,
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
                triangle: feature_triangle(point.feature),
            })
            .collect(),
        step,
    }
}
