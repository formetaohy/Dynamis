use super::FrameParams;
use super::dispatch::POSITION_SOLVE_EXTRACT;
use super::stage::{RO, RW, Stage, UNIFORM, whole};
use crate::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_CONTACTS;

pub(super) struct PositionSolve {
    position_solve_extract: Stage,
    body_apply_positions: Stage,
}

impl PositionSolve {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            position_solve_extract: Stage::build(
                context,
                "position_solve_extract",
                include_str!("../shaders/position_solve_extract.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RO, whole(&buffers.contacts.manifolds)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RW, whole(&buffers.contacts.deltas)),
                ],
                &[],
            ),
            body_apply_positions: Stage::build(
                context,
                "body_apply_positions",
                include_str!("../shaders/body_apply_positions.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.contacts.first_a)),
                    (RO, whole(&buffers.contacts.first_b)),
                    (RO, whole(&buffers.contacts.a_body)),
                    (RO, whole(&buffers.contacts.b_keys)),
                    (RO, whole(&buffers.contacts.b_values)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RO, whole(&buffers.contacts.deltas)),
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
        for _ in 0..params.position_iterations {
            self.position_solve_extract.record_indirect(
                recorder,
                &buffers.dispatch,
                POSITION_SOLVE_EXTRACT,
            );
            self.body_apply_positions
                .record(recorder, params.dynamic_count);
        }
    }
}
