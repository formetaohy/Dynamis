use super::FrameParams;
use super::stage::{RO, RW, Stage, UNIFORM, whole};
use crate::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_COUNT, COUNTER_JOINTS};

pub(super) struct Commands {
    reset_counters: Stage,
    body_gather: Stage,
    body_scatter: Stage,
    body_edits: Stage,
    constraint_gather: Stage,
    constraint_scatter: Stage,
    joint_filter: Stage,
}

impl Commands {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            reset_counters: Stage::build(
                context,
                "reset_counters",
                include_str!("../shaders/reset_counters.wgsl"),
                per_row,
                &[(RW, whole(&buffers.counters))],
                &[],
            ),
            body_gather: Stage::build(
                context,
                "body_gather",
                include_str!("../shaders/body_gather.wgsl"),
                per_row,
                &[
                    (RO, whole(&buffers.bodies.states)),
                    (RW, whole(&buffers.bodies.state_scratch)),
                    (RO, whole(&buffers.bodies.row_src)),
                    (RO, whole(&buffers.bodies.row_fresh)),
                    (RO, whole(&buffers.bodies.fresh_states)),
                    (UNIFORM, whole(&buffers.params)),
                ],
                &[],
            ),
            body_scatter: Stage::build(
                context,
                "body_scatter",
                include_str!("../shaders/body_scatter.wgsl"),
                per_row,
                &[
                    (RW, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.state_scratch)),
                    (UNIFORM, whole(&buffers.params)),
                ],
                &[],
            ),
            body_edits: Stage::build(
                context,
                "body_edits",
                include_str!("../shaders/body_edits.wgsl"),
                per_row,
                &[
                    (RO, whole(&buffers.bodies.commands)),
                    (RO, whole(&buffers.bodies.command_first)),
                    (RW, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RW, whole(&buffers.islands.wake_flags)),
                    (UNIFORM, whole(&buffers.params)),
                ],
                &[],
            ),
            constraint_gather: Stage::build(
                context,
                "constraint_gather",
                include_str!("../shaders/constraint_gather.wgsl"),
                per_row,
                &[
                    (RO, whole(&buffers.constraints.runtime)),
                    (RW, whole(&buffers.constraints.scratch)),
                    (RO, whole(&buffers.constraints.row_src)),
                    (RO, whole(&buffers.constraints.row_fresh)),
                    (RO, whole(&buffers.constraints.fresh)),
                    (UNIFORM, whole(&buffers.params)),
                ],
                &[],
            ),
            constraint_scatter: Stage::build(
                context,
                "constraint_scatter",
                include_str!("../shaders/constraint_scatter.wgsl"),
                per_row,
                &[
                    (RW, whole(&buffers.constraints.runtime)),
                    (RO, whole(&buffers.constraints.scratch)),
                    (UNIFORM, whole(&buffers.params)),
                ],
                &[],
            ),
            joint_filter: Stage::build(
                context,
                "joint_filter",
                include_str!("../shaders/joint_filter.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.constraints.descriptors)),
                    (RO, whole(&buffers.constraints.runtime)),
                    (RW, whole(&buffers.constraints.joint_hi)),
                    (RW, whole(&buffers.constraints.joint_lo)),
                    (RW, buffers.counter(COUNTER_JOINTS)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn reset(&self, recorder: &mut ComputeRecorder) {
        self.reset_counters.record(recorder, COUNTER_COUNT as u32);
    }

    /// Structural row moves and per-slot edits, shared by the step and an out-of-step
    /// query flush.
    pub(super) fn record_moves(&self, recorder: &mut ComputeRecorder, params: &FrameParams) {
        if params.body_structural {
            self.body_gather.record(recorder, params.body_count);
            self.body_scatter.record(recorder, params.body_count);
        }
        if params.has_body_edits {
            self.body_edits.record(recorder, params.body_count);
        }
        if params.constraint_structural {
            self.constraint_gather
                .record(recorder, params.constraint_count);
            self.constraint_scatter
                .record(recorder, params.constraint_count);
        }
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, params: &FrameParams) {
        self.reset(recorder);
        self.record_moves(recorder, params);
        if params.constraint_count > 0 {
            self.joint_filter.record(recorder, params.constraint_count);
        }
    }
}
