use super::World;
use super::backend::segment::{Arrival, SegmentKind};
use super::device::Facts;
use dynamis_abi::COUNTER_RESTING;
use dynamis_abi::{
    COUNTER_CONTACTS, COUNTER_DEVICE_COUNT, COUNTER_SOLVE_ANGULAR_RESIDUAL,
    COUNTER_SOLVE_LINEAR_RESIDUAL, COUNTER_STRIDE, ConstraintReactionRecord, ContactRecord,
    Counters, DeclaredCounters, FEATURE_KIND_MASK, FEATURE_TRIANGLE, FEATURE_TRIANGLE_MASK,
    JointStateRecord, NO_SURFACE, SHAPE_HEIGHTFIELD, SHAPE_MESH, solve_velocity,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveResidual {
    pub linear: f32,
    pub angular: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub slot: usize,
    pub counter: &'static str,
    pub label: &'static str,
    pub shortfall: dynamis_abi::Shortfall,
    pub count: u32,
}

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

    pub(crate) fn consume_pack(&mut self, step: u64, declared: DeclaredCounters, bytes: &[u8]) {
        declared.write_into(&mut self.backend.measured);
        measured_counters(bytes, &mut self.backend.measured);
        assert_eq!(
            self.backend.measured[dynamis_abi::COUNTER_STEP],
            step as u32 + 1,
            "the device must close exactly the step the host declared"
        );
        self.accept_measured(step);
    }

    pub fn inspect_contacts(&mut self) -> Vec<ContactManifold> {
        self.sync(Facts::Landed);
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
            regions.push((active_buffer, 0, active as u64 * CONTACT_BYTES));
        }
        if resting > 0 {
            regions.push((resting_live, 0, resting as u64 * 4));
            regions.push((resting_buffer, 0, resting as u64 * CONTACT_BYTES));
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

    pub(crate) fn collect_readbacks(&mut self) {
        self.backend.gpu.poll();
        self.collect_observations();
        let arrivals = self.backend.segments.collect();
        self.consume_segments(arrivals);
        for ((step, declared), bytes) in self.backend.readback.counters.collect() {
            self.consume_pack(step, declared, &bytes);
        }
        for (batch, bytes) in self.backend.readback.queries.collect() {
            self.collect_query_batch(batch, &bytes);
        }
        #[cfg(feature = "profile")]
        for timings in self.backend.passes.collect_timings() {
            self.backend.pass_timings = timings;
        }
    }

    pub(crate) fn retire_fact_buffers(&mut self) {
        self.drain_observations();
        for ((step, declared), bytes) in self.backend.readback.counters.drain() {
            self.consume_pack(step, declared, &bytes);
        }
        for (batch, bytes) in self.backend.readback.queries.drain() {
            self.collect_query_batch(batch, &bytes);
        }
        #[cfg(feature = "profile")]
        for timings in self.backend.passes.collect_timings() {
            self.backend.pass_timings = timings;
        }
    }

    pub(crate) fn consume_segments(&mut self, arrivals: Vec<Arrival>) {
        for arrival in arrivals {
            match arrival.kind {
                SegmentKind::Events | SegmentKind::SoftEvents => {
                    self.consume_events(arrival.count, &arrival.bytes)
                }
                SegmentKind::Impacts => {
                    self.consume_impacts(arrival.step, arrival.count, &arrival.bytes)
                }
                SegmentKind::Breaks => self.consume_breaks(arrival.count, &arrival.bytes),
            }
        }
    }

    pub(crate) fn accept_measured(&mut self, step: u64) {
        self.assert_no_device_faults();
        self.backend.measured_step = Some(step);
        self.backend.segments.close(&self.backend.measured, step);
    }

    fn assert_no_device_faults(&self) {
        for (slot, counter) in dynamis_abi::COUNTERS.iter().enumerate() {
            if !counter.is_fatal() {
                continue;
            }
            let refused = self.backend.measured[slot];
            assert_eq!(
                refused, 0,
                "the device dropped {refused} {} ({}) that a step cannot lose",
                counter.label, counter.name,
            );
        }
    }

    pub fn measured(&self) -> &Counters {
        &self.backend.measured
    }

    pub fn solve_residual(&self) -> SolveResidual {
        SolveResidual {
            linear: solve_velocity(self.backend.measured[COUNTER_SOLVE_LINEAR_RESIDUAL]),
            angular: solve_velocity(self.backend.measured[COUNTER_SOLVE_ANGULAR_RESIDUAL]),
        }
    }

    pub fn refusals(&self) -> Vec<Refusal> {
        dynamis_abi::COUNTERS
            .iter()
            .enumerate()
            .filter(|(_, counter)| counter.refuses())
            .filter_map(|(slot, counter)| {
                let count = self.backend.measured[slot];
                (count > 0).then_some(Refusal {
                    slot,
                    counter: counter.name,
                    label: counter.label,
                    shortfall: counter.shortfall,
                    count,
                })
            })
            .collect()
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
        let handle = ConstraintHandle {
            id: constraint_id,
            generation,
        };
        if !self.constraints.is_alive(handle) {
            return;
        }
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
            contact_frequency: dynamis_abi::contact_frequency(record.relaxation),
            contact_damping_ratio: record.damping_ratio,
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
