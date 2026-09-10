use super::FrameParams;
use super::dispatch::CONTACT_SOLVE_EXTRACT;
use super::stage::{RO, RW, Stage, UNIFORM, whole};
use crate::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_CONTACTS;

pub(super) struct VelocitySolve {
    contact_solve_extract: Stage,
    constraint_solve_extract: Stage,
    body_apply_solver: Stage,
}

impl VelocitySolve {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            contact_solve_extract: Stage::build(
                context,
                "contact_solve_extract",
                include_str!("../shaders/contact_solve_extract.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RW, whole(&buffers.contacts.manifolds)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RW, whole(&buffers.islands.wake_flags)),
                    (RW, whole(&buffers.contacts.deltas)),
                ],
                &[],
            ),
            constraint_solve_extract: Stage::build(
                context,
                "constraint_solve_extract",
                include_str!("../shaders/constraint_solve_extract.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RO, whole(&buffers.constraints.descriptors)),
                    (RW, whole(&buffers.constraints.runtime)),
                    (RW, whole(&buffers.islands.wake_flags)),
                    (RW, whole(&buffers.constraints.deltas)),
                ],
                &[],
            ),
            body_apply_solver: Stage::build(
                context,
                "body_apply_solver",
                include_str!("../shaders/body_apply_solver.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.contacts.first_a)),
                    (RO, whole(&buffers.contacts.first_b)),
                    (RO, whole(&buffers.contacts.a_body)),
                    (RO, whole(&buffers.contacts.b_keys)),
                    (RO, whole(&buffers.contacts.b_values)),
                    (RO, whole(&buffers.constraints.first_a)),
                    (RO, whole(&buffers.constraints.first_b)),
                    (RO, whole(&buffers.constraints.a_keys)),
                    (RO, whole(&buffers.constraints.a_values)),
                    (RO, whole(&buffers.constraints.b_keys)),
                    (RO, whole(&buffers.constraints.b_values)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RO, whole(&buffers.contacts.deltas)),
                    (RO, whole(&buffers.constraints.deltas)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        params: &FrameParams,
    ) {
        let constraint_active = params.constraint_count > 0;
        for _ in 0..params.solve_iterations {
            if constraint_active {
                self.constraint_solve_extract
                    .record(recorder, params.constraint_count);
            }
            self.contact_solve_extract.record_indirect(
                recorder,
                &buffers.dispatch,
                CONTACT_SOLVE_EXTRACT,
            );
            self.body_apply_solver
                .record(recorder, params.dynamic_count);
        }
    }
}
