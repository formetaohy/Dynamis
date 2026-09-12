use super::FrameParams;
use super::stage::{CORE, Stage, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_ACTIVE, COUNTER_COUNT, COUNTER_JOINTS, COUNTER_SLEPT, COUNTER_WOKE};

pub(super) struct Commands {
    reset_counters: Stage,
    body_move_gather: Stage,
    body_move_scatter: Stage,
    body_edits: Stage,
    row_of_body: Stage,
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
                include_str!("shaders/reset_counters.wgsl"),
                per_row,
                CORE,
                &[("counters", whole(&buffers.counters))],
                &[],
            ),
            body_move_gather: Stage::build(
                context,
                "body_move_gather",
                include_str!("shaders/body_move_gather.wgsl"),
                per_row,
                CORE,
                &[
                    ("body_states", whole(&buffers.body_states)),
                    ("state_scratch", whole(&buffers.body_state_scratch)),
                    ("row_moves", whole(&buffers.body_row_moves)),
                    ("fresh_rows", whole(&buffers.body_fresh_rows)),
                    ("params", whole(&buffers.params)),
                ],
                &[],
            ),
            body_move_scatter: Stage::build(
                context,
                "body_move_scatter",
                include_str!("shaders/body_move_scatter.wgsl"),
                per_row,
                CORE,
                &[
                    ("body_states", whole(&buffers.body_states)),
                    ("state_scratch", whole(&buffers.body_state_scratch)),
                    ("row_moves", whole(&buffers.body_row_moves)),
                    ("params", whole(&buffers.params)),
                ],
                &[],
            ),
            body_edits: Stage::build(
                context,
                "body_edits",
                include_str!("shaders/body_edits.wgsl"),
                per_row,
                CORE,
                &[
                    ("edits", whole(&buffers.body_edits)),
                    ("edit_runs", whole(&buffers.body_edit_runs)),
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("wake_flags", whole(&buffers.wake_flags)),
                    ("params", whole(&buffers.params)),
                    ("slept_count", buffers.counter(COUNTER_SLEPT)),
                    ("woke_count", buffers.counter(COUNTER_WOKE)),
                ],
                &[],
            ),
            row_of_body: Stage::build(
                context,
                "row_of_body",
                include_str!("shaders/row_of_body.wgsl"),
                per_row,
                CORE,
                &[
                    ("body_states", whole(&buffers.body_states)),
                    ("row_moves", whole(&buffers.body_row_moves)),
                    ("row_of_body", whole(&buffers.body_row_of_id)),
                    ("params", whole(&buffers.params)),
                ],
                &[],
            ),
            constraint_move_gather: Stage::build(
                context,
                "constraint_move_gather",
                include_str!("shaders/constraint_move_gather.wgsl"),
                per_row,
                CORE,
                &[
                    ("constraint_runtime", whole(&buffers.constraint_runtime)),
                    ("constraint_scratch", whole(&buffers.constraint_scratch)),
                    ("row_moves", whole(&buffers.constraint_row_moves)),
                    ("fresh_rows", whole(&buffers.constraint_fresh_rows)),
                    ("params", whole(&buffers.params)),
                ],
                &[],
            ),
            constraint_move_scatter: Stage::build(
                context,
                "constraint_move_scatter",
                include_str!("shaders/constraint_move_scatter.wgsl"),
                per_row,
                CORE,
                &[
                    ("constraint_runtime", whole(&buffers.constraint_runtime)),
                    ("constraint_scratch", whole(&buffers.constraint_scratch)),
                    ("row_moves", whole(&buffers.constraint_row_moves)),
                    ("params", whole(&buffers.params)),
                ],
                &[],
            ),
            activity: Stage::build(
                context,
                "activity",
                include_str!("shaders/activity.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("body_activity", whole(&buffers.body_activity)),
                    ("active_count", buffers.counter(COUNTER_ACTIVE)),
                ],
                &[],
            ),
            joint_filter: Stage::build(
                context,
                "joint_filter",
                include_str!("shaders/joint_filter.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("constraint_descs", whole(&buffers.constraint_descriptors)),
                    ("constraint_runtime", whole(&buffers.constraint_runtime)),
                    ("joint_major", whole(&buffers.joint_filter_major)),
                    ("joint_minor", whole(&buffers.joint_filter_minor)),
                    ("joint_count", buffers.counter(COUNTER_JOINTS)),
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
            self.row_of_body.record(recorder, params.body_move_count);
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
