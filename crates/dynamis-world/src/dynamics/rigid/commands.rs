use super::Count;
use super::Frame;
use super::buffers::{RigidBuffers, StreamId};
use super::shader;
use super::shader::CORE;
use crate::dynamics::engine::{Stage, workgroups_of};
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_ACTIVE, COUNTER_COUNT, COUNTER_JOINTS, COUNTER_SLEPT, COUNTER_WOKE};

pub(super) struct Commands {
    reset_counters: Stage,
    body_move_gather: Stage,
    body_move_scatter: Stage,
    body_edits: Stage,
    row_of_body: Stage,
    constraint_rows: Stage,
    constraint_move_gather: Stage,
    constraint_move_scatter: Stage,
    joint_filter: Stage,
    activity: Stage,
}

impl Commands {
    pub(super) fn build(context: &GpuContext, buffers: &RigidBuffers) -> Self {
        Self {
            reset_counters: Stage::build(
                context,
                "reset_counters",
                shader::workgroups(context, include_str!("shaders/reset_counters.wgsl"), CORE),
                buffers,
                &[("counters", StreamId::Counters.whole())],
                &[],
            ),
            body_move_gather: Stage::build(
                context,
                "body_move_gather",
                shader::rows(
                    context,
                    include_str!("shaders/body_move_gather.wgsl"),
                    CORE,
                    Count::BodyMoves,
                ),
                buffers,
                &[
                    ("body_states", StreamId::BodyStates.whole()),
                    ("state_scratch", StreamId::BodyStateScratch.whole()),
                    ("row_moves", StreamId::BodyRowMoves.whole()),
                    ("fresh_rows", StreamId::BodyFreshRows.whole()),
                    ("params", StreamId::Params.whole()),
                ],
                &[],
            ),
            body_move_scatter: Stage::build(
                context,
                "body_move_scatter",
                shader::rows(
                    context,
                    include_str!("shaders/body_move_scatter.wgsl"),
                    CORE,
                    Count::BodyMoves,
                ),
                buffers,
                &[
                    ("body_states", StreamId::BodyStates.whole()),
                    ("state_scratch", StreamId::BodyStateScratch.whole()),
                    ("row_moves", StreamId::BodyRowMoves.whole()),
                    ("params", StreamId::Params.whole()),
                ],
                &[],
            ),
            body_edits: Stage::build(
                context,
                "body_edits",
                shader::rows(
                    context,
                    include_str!("shaders/body_edits.wgsl"),
                    CORE,
                    Count::EditRuns,
                ),
                buffers,
                &[
                    ("edits", StreamId::BodyEdits.whole()),
                    ("edit_runs", StreamId::BodyEditRuns.whole()),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("body_descs", StreamId::BodyDescriptors.whole()),
                    ("wake_flags", StreamId::WakeFlags.whole()),
                    ("params", StreamId::Params.whole()),
                    ("slept_count", buffers.counter(COUNTER_SLEPT)),
                    ("woke_count", buffers.counter(COUNTER_WOKE)),
                ],
                &[],
            ),
            row_of_body: Stage::build(
                context,
                "row_of_body",
                shader::rows(
                    context,
                    include_str!("shaders/row_of_body.wgsl"),
                    CORE,
                    Count::BodyMoves,
                ),
                buffers,
                &[
                    ("body_states", StreamId::BodyStates.whole()),
                    ("row_moves", StreamId::BodyRowMoves.whole()),
                    ("row_of_body", StreamId::BodyRowOfId.whole()),
                    ("params", StreamId::Params.whole()),
                ],
                &[],
            ),
            constraint_rows: Stage::build(
                context,
                "constraint_rows",
                shader::rows(
                    context,
                    include_str!("shaders/constraint_rows.wgsl"),
                    CORE,
                    Count::Constraints,
                ),
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("constraint_descs", StreamId::ConstraintDescriptors.whole()),
                    ("row_of_body", StreamId::BodyRowOfId.whole()),
                    ("constraint_rows", StreamId::ConstraintRows.whole()),
                ],
                &[],
            ),
            constraint_move_gather: Stage::build(
                context,
                "constraint_move_gather",
                shader::rows(
                    context,
                    include_str!("shaders/constraint_move_gather.wgsl"),
                    CORE,
                    Count::ConstraintMoves,
                ),
                buffers,
                &[
                    ("constraint_runtime", StreamId::ConstraintRuntime.whole()),
                    ("constraint_scratch", StreamId::ConstraintScratch.whole()),
                    ("row_moves", StreamId::ConstraintRowMoves.whole()),
                    ("fresh_rows", StreamId::ConstraintFreshRows.whole()),
                    ("params", StreamId::Params.whole()),
                ],
                &[],
            ),
            constraint_move_scatter: Stage::build(
                context,
                "constraint_move_scatter",
                shader::rows(
                    context,
                    include_str!("shaders/constraint_move_scatter.wgsl"),
                    CORE,
                    Count::ConstraintMoves,
                ),
                buffers,
                &[
                    ("constraint_runtime", StreamId::ConstraintRuntime.whole()),
                    ("constraint_scratch", StreamId::ConstraintScratch.whole()),
                    ("row_moves", StreamId::ConstraintRowMoves.whole()),
                    ("params", StreamId::Params.whole()),
                ],
                &[],
            ),
            activity: Stage::build(
                context,
                "activity",
                shader::rows(
                    context,
                    include_str!("shaders/activity.wgsl"),
                    CORE,
                    Count::Bodies,
                ),
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("body_descs", StreamId::BodyDescriptors.whole()),
                    ("body_activity", StreamId::BodyActivity.whole()),
                    ("active_count", buffers.counter(COUNTER_ACTIVE)),
                ],
                &[],
            ),
            joint_filter: Stage::build(
                context,
                "joint_filter",
                shader::rows(
                    context,
                    include_str!("shaders/joint_filter.wgsl"),
                    CORE,
                    Count::Constraints,
                ),
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("constraint_descs", StreamId::ConstraintDescriptors.whole()),
                    ("constraint_runtime", StreamId::ConstraintRuntime.whole()),
                    ("joint_major", StreamId::JointFilterMajor.whole()),
                    ("joint_minor", StreamId::JointFilterMinor.whole()),
                    ("joint_count", buffers.counter(COUNTER_JOINTS)),
                    ("constraint_rows", StreamId::ConstraintRows.whole()),
                ],
                &[],
            ),
        }
    }

    pub(super) fn reset(&self, recorder: &mut ComputeRecorder, buffers: &RigidBuffers) {
        self.reset_counters.record_workgroups(
            recorder,
            buffers,
            workgroups_of(COUNTER_COUNT as u32),
        );
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        self.reset(recorder, buffers);
        self.record_moves(recorder, buffers, frame);
        self.record_edits(recorder, buffers, frame);
        self.constraint_rows
            .record_rows(recorder, buffers, Count::Constraints.rows(&frame.params));
        self.joint_filter
            .record_rows(recorder, buffers, Count::Constraints.rows(&frame.params));
        self.activity
            .record_rows(recorder, buffers, Count::Bodies.rows(&frame.params));
    }

    pub(super) fn record_moves(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        self.body_move_gather
            .record_rows(recorder, buffers, Count::BodyMoves.rows(&frame.params));
        self.body_move_scatter
            .record_rows(recorder, buffers, Count::BodyMoves.rows(&frame.params));
        self.row_of_body
            .record_rows(recorder, buffers, Count::BodyMoves.rows(&frame.params));
        self.constraint_move_gather.record_rows(
            recorder,
            buffers,
            Count::ConstraintMoves.rows(&frame.params),
        );
        self.constraint_move_scatter.record_rows(
            recorder,
            buffers,
            Count::ConstraintMoves.rows(&frame.params),
        );
    }

    pub(super) fn record_edits(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        self.body_edits
            .record_rows(recorder, buffers, Count::EditRuns.rows(&frame.params));
    }
}
