use super::FrameParams;
use super::stage::{CORE, RO, RW, Stage, UNIFORM, whole};
use crate::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_ACTIVE, COUNTER_COUNT, COUNTER_JOINTS, COUNTER_SLEPT, COUNTER_WOKE};

pub(super) struct Commands {
    reset_counters: Stage,
    body_move_gather: Stage,
    body_move_scatter: Stage,
    body_edits: Stage,
    body_rows: Stage,
    constraint_move_gather: Stage,
    constraint_move_scatter: Stage,
    joint_filter: Stage,
    activity: Stage,
}

impl Commands {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            reset_counters: Stage::build(
                context,
                "reset_counters",
                include_str!("../shaders/reset_counters.wgsl"),
                per_row,
                CORE,
                &[(RW, whole(&buffers.counters))],
                &[],
            ),
            body_move_gather: Stage::build(
                context,
                "body_move_gather",
                include_str!("../shaders/body_move_gather.wgsl"),
                per_row,
                CORE,
                &[
                    (RO, whole(&buffers.bodies.states)),
                    (RW, whole(&buffers.bodies.state_scratch)),
                    (RO, whole(&buffers.bodies.row_moves)),
                    (RO, whole(&buffers.bodies.fresh_rows)),
                    (UNIFORM, whole(&buffers.params)),
                ],
                &[],
            ),
            body_move_scatter: Stage::build(
                context,
                "body_move_scatter",
                include_str!("../shaders/body_move_scatter.wgsl"),
                per_row,
                CORE,
                &[
                    (RW, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.state_scratch)),
                    (RO, whole(&buffers.bodies.row_moves)),
                    (UNIFORM, whole(&buffers.params)),
                ],
                &[],
            ),
            body_edits: Stage::build(
                context,
                "body_edits",
                include_str!("../shaders/body_edits.wgsl"),
                per_row,
                CORE,
                &[
                    (RO, whole(&buffers.bodies.edits)),
                    (RO, whole(&buffers.bodies.edit_runs)),
                    (RW, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RW, whole(&buffers.islands.wake_flags)),
                    (UNIFORM, whole(&buffers.params)),
                    (RW, buffers.counter(COUNTER_SLEPT)),
                    (RW, buffers.counter(COUNTER_WOKE)),
                ],
                &[],
            ),
            body_rows: Stage::build(
                context,
                "body_rows",
                include_str!("../shaders/body_rows.wgsl"),
                per_row,
                CORE,
                &[
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.row_moves)),
                    (RW, whole(&buffers.bodies.rows)),
                    (UNIFORM, whole(&buffers.params)),
                ],
                &[],
            ),
            constraint_move_gather: Stage::build(
                context,
                "constraint_move_gather",
                include_str!("../shaders/constraint_move_gather.wgsl"),
                per_row,
                CORE,
                &[
                    (RO, whole(&buffers.constraints.runtime)),
                    (RW, whole(&buffers.constraints.scratch)),
                    (RO, whole(&buffers.constraints.row_moves)),
                    (RO, whole(&buffers.constraints.fresh_rows)),
                    (UNIFORM, whole(&buffers.params)),
                ],
                &[],
            ),
            constraint_move_scatter: Stage::build(
                context,
                "constraint_move_scatter",
                include_str!("../shaders/constraint_move_scatter.wgsl"),
                per_row,
                CORE,
                &[
                    (RW, whole(&buffers.constraints.runtime)),
                    (RO, whole(&buffers.constraints.scratch)),
                    (RO, whole(&buffers.constraints.row_moves)),
                    (UNIFORM, whole(&buffers.params)),
                ],
                &[],
            ),
            activity: Stage::build(
                context,
                "activity",
                include_str!("../shaders/activity.wgsl"),
                per_row,
                CORE,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RW, whole(&buffers.bodies.activity)),
                    (RW, buffers.counter(COUNTER_ACTIVE)),
                ],
                &[],
            ),
            joint_filter: Stage::build(
                context,
                "joint_filter",
                include_str!("../shaders/joint_filter.wgsl"),
                per_row,
                CORE,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.constraints.descriptors)),
                    (RO, whole(&buffers.constraints.runtime)),
                    (RW, whole(&buffers.constraints.joint_major)),
                    (RW, whole(&buffers.constraints.joint_minor)),
                    (RW, buffers.counter(COUNTER_JOINTS)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn reset(&self, recorder: &mut ComputeRecorder) {
        self.reset_counters.record(recorder, COUNTER_COUNT as u32);
    }

    pub(super) fn record_moves(&self, recorder: &mut ComputeRecorder, params: &FrameParams) {
        if params.body_move_count > 0 {
            self.body_move_gather
                .record(recorder, params.body_move_count);
            self.body_move_scatter
                .record(recorder, params.body_move_count);
            self.body_rows.record(recorder, params.body_move_count);
        }
        if params.constraint_move_count > 0 {
            self.constraint_move_gather
                .record(recorder, params.constraint_move_count);
            self.constraint_move_scatter
                .record(recorder, params.constraint_move_count);
        }
    }

    pub(super) fn record_edits(&self, recorder: &mut ComputeRecorder, edit_run_count: u32) {
        self.body_edits.record(recorder, edit_run_count);
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, params: &FrameParams) {
        self.reset(recorder);
        self.record_moves(recorder, params);
        self.record_edits(recorder, params.edit_run_count);
        if params.constraint_count > 0 {
            self.joint_filter.record(recorder, params.constraint_count);
        }
        self.activity.record(recorder, params.body_count);
    }
}
