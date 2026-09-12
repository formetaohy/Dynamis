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
                    ("body_states", whole(&buffers.bodies.states)),
                    ("state_scratch", whole(&buffers.bodies.state_scratch)),
                    ("row_moves", whole(&buffers.bodies.row_moves)),
                    ("fresh_rows", whole(&buffers.bodies.fresh_rows)),
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
                    ("body_states", whole(&buffers.bodies.states)),
                    ("state_scratch", whole(&buffers.bodies.state_scratch)),
                    ("row_moves", whole(&buffers.bodies.row_moves)),
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
                    ("edits", whole(&buffers.bodies.edits)),
                    ("edit_runs", whole(&buffers.bodies.edit_runs)),
                    ("body_states", whole(&buffers.bodies.states)),
                    ("body_descs", whole(&buffers.bodies.descriptors)),
                    ("wake_flags", whole(&buffers.islands.wake_flags)),
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
                    ("body_states", whole(&buffers.bodies.states)),
                    ("row_moves", whole(&buffers.bodies.row_moves)),
                    ("row_of_body", whole(&buffers.bodies.row_of_body)),
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
                    ("constraint_runtime", whole(&buffers.constraints.runtime)),
                    ("constraint_scratch", whole(&buffers.constraints.scratch)),
                    ("row_moves", whole(&buffers.constraints.row_moves)),
                    ("fresh_rows", whole(&buffers.constraints.fresh_rows)),
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
                    ("constraint_runtime", whole(&buffers.constraints.runtime)),
                    ("constraint_scratch", whole(&buffers.constraints.scratch)),
                    ("row_moves", whole(&buffers.constraints.row_moves)),
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
                    ("body_states", whole(&buffers.bodies.states)),
                    ("body_descs", whole(&buffers.bodies.descriptors)),
                    ("body_activity", whole(&buffers.bodies.activity)),
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
                    ("constraint_descs", whole(&buffers.constraints.descriptors)),
                    ("constraint_runtime", whole(&buffers.constraints.runtime)),
                    ("joint_major", whole(&buffers.constraints.joint_major)),
                    ("joint_minor", whole(&buffers.constraints.joint_minor)),
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
