use super::Count;
use super::streams::RigidStream;
use crate::dynamics::Frame;
use crate::dynamics::engine::{Stage, workgroups_of};
use crate::dynamics::scene::SceneStream;
use crate::dynamics::shader;
use crate::dynamics::shader::CORE;
use crate::dynamics::streams::Streams;
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
    pub(super) fn build(context: &GpuContext, streams: &Streams) -> Self {
        Self {
            reset_counters: Stage::build(
                context,
                "reset_counters",
                shader::workgroups(
                    context,
                    include_str!("../shaders/reset_counters.wgsl"),
                    CORE,
                ),
                streams,
                &[("counters", SceneStream::Counters.whole())],
                &[],
            ),
            body_move_gather: Stage::build(
                context,
                "body_move_gather",
                shader::rows(
                    context,
                    include_str!("../shaders/body_move_gather.wgsl"),
                    CORE,
                    Count::BodyMoves,
                ),
                streams,
                &[
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("state_scratch", RigidStream::BodyStateScratch.whole()),
                    ("row_moves", SceneStream::BodyRowMoves.whole()),
                    ("fresh_rows", SceneStream::BodyFreshRows.whole()),
                    ("params", SceneStream::Params.whole()),
                ],
                &[],
            ),
            body_move_scatter: Stage::build(
                context,
                "body_move_scatter",
                shader::rows(
                    context,
                    include_str!("../shaders/body_move_scatter.wgsl"),
                    CORE,
                    Count::BodyMoves,
                ),
                streams,
                &[
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("state_scratch", RigidStream::BodyStateScratch.whole()),
                    ("row_moves", SceneStream::BodyRowMoves.whole()),
                    ("params", SceneStream::Params.whole()),
                ],
                &[],
            ),
            body_edits: Stage::build(
                context,
                "body_edits",
                shader::rows(
                    context,
                    include_str!("../shaders/body_edits.wgsl"),
                    CORE,
                    Count::EditRuns,
                ),
                streams,
                &[
                    ("edits", SceneStream::BodyEdits.whole()),
                    ("edit_runs", SceneStream::BodyEditRuns.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("wake_flags", RigidStream::WakeFlags.whole()),
                    ("params", SceneStream::Params.whole()),
                    ("slept_count", streams.scene.counter(COUNTER_SLEPT)),
                    ("woke_count", streams.scene.counter(COUNTER_WOKE)),
                ],
                &[],
            ),
            row_of_body: Stage::build(
                context,
                "row_of_body",
                shader::rows(
                    context,
                    include_str!("../shaders/row_of_body.wgsl"),
                    CORE,
                    Count::BodyMoves,
                ),
                streams,
                &[
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("row_moves", SceneStream::BodyRowMoves.whole()),
                    ("row_of_body", SceneStream::BodyRowOfId.whole()),
                    ("params", SceneStream::Params.whole()),
                ],
                &[],
            ),
            constraint_rows: Stage::build(
                context,
                "constraint_rows",
                shader::rows(
                    context,
                    include_str!("../shaders/constraint_rows.wgsl"),
                    CORE,
                    Count::Constraints,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    (
                        "constraint_descs",
                        SceneStream::ConstraintDescriptors.whole(),
                    ),
                    ("row_of_body", SceneStream::BodyRowOfId.whole()),
                    ("constraint_rows", RigidStream::ConstraintRows.whole()),
                ],
                &[],
            ),
            constraint_move_gather: Stage::build(
                context,
                "constraint_move_gather",
                shader::rows(
                    context,
                    include_str!("../shaders/constraint_move_gather.wgsl"),
                    CORE,
                    Count::ConstraintMoves,
                ),
                streams,
                &[
                    ("constraint_runtime", SceneStream::ConstraintRuntime.whole()),
                    ("constraint_scratch", RigidStream::ConstraintScratch.whole()),
                    ("row_moves", SceneStream::ConstraintRowMoves.whole()),
                    ("fresh_rows", SceneStream::ConstraintFreshRows.whole()),
                    ("params", SceneStream::Params.whole()),
                ],
                &[],
            ),
            constraint_move_scatter: Stage::build(
                context,
                "constraint_move_scatter",
                shader::rows(
                    context,
                    include_str!("../shaders/constraint_move_scatter.wgsl"),
                    CORE,
                    Count::ConstraintMoves,
                ),
                streams,
                &[
                    ("constraint_runtime", SceneStream::ConstraintRuntime.whole()),
                    ("constraint_scratch", RigidStream::ConstraintScratch.whole()),
                    ("row_moves", SceneStream::ConstraintRowMoves.whole()),
                    ("params", SceneStream::Params.whole()),
                ],
                &[],
            ),
            activity: Stage::build(
                context,
                "activity",
                shader::rows(
                    context,
                    include_str!("../shaders/activity.wgsl"),
                    CORE,
                    Count::Bodies,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("body_activity", RigidStream::BodyActivity.whole()),
                    ("active_count", streams.scene.counter(COUNTER_ACTIVE)),
                ],
                &[],
            ),
            joint_filter: Stage::build(
                context,
                "joint_filter",
                shader::rows(
                    context,
                    include_str!("../shaders/joint_filter.wgsl"),
                    CORE,
                    Count::Constraints,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    (
                        "constraint_descs",
                        SceneStream::ConstraintDescriptors.whole(),
                    ),
                    ("constraint_runtime", SceneStream::ConstraintRuntime.whole()),
                    ("joint_major", RigidStream::JointFilterMajor.whole()),
                    ("joint_minor", RigidStream::JointFilterMinor.whole()),
                    ("joint_count", streams.scene.counter(COUNTER_JOINTS)),
                    ("constraint_rows", RigidStream::ConstraintRows.whole()),
                ],
                &[],
            ),
        }
    }

    pub(super) fn reset(&self, recorder: &mut ComputeRecorder, streams: &Streams) {
        self.reset_counters.record_workgroups(
            recorder,
            streams,
            workgroups_of(COUNTER_COUNT as u32),
        );
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, streams: &Streams, frame: &Frame) {
        self.reset(recorder, streams);
        self.record_moves(recorder, streams, frame);
        self.record_edits(recorder, streams, frame);
        self.constraint_rows
            .record_rows(recorder, streams, Count::Constraints.rows(&frame.params));
        self.joint_filter
            .record_rows(recorder, streams, Count::Constraints.rows(&frame.params));
        self.activity
            .record_rows(recorder, streams, Count::Bodies.rows(&frame.params));
    }

    pub(super) fn record_moves(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &Streams,
        frame: &Frame,
    ) {
        self.body_move_gather
            .record_rows(recorder, streams, Count::BodyMoves.rows(&frame.params));
        self.body_move_scatter
            .record_rows(recorder, streams, Count::BodyMoves.rows(&frame.params));
        self.row_of_body
            .record_rows(recorder, streams, Count::BodyMoves.rows(&frame.params));
        self.constraint_move_gather.record_rows(
            recorder,
            streams,
            Count::ConstraintMoves.rows(&frame.params),
        );
        self.constraint_move_scatter.record_rows(
            recorder,
            streams,
            Count::ConstraintMoves.rows(&frame.params),
        );
    }

    pub(super) fn record_edits(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &Streams,
        frame: &Frame,
    ) {
        self.body_edits
            .record_rows(recorder, streams, Count::EditRuns.rows(&frame.params));
    }
}
