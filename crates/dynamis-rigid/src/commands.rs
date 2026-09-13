use super::streams::RigidStream;
use dynamis_abi::{COUNTER_ACTIVE, COUNTER_COUNT, COUNTER_JOINTS, COUNTER_SLEPT, COUNTER_WOKE};
use dynamis_engine::Resources;
use dynamis_engine::{Stage, workgroups_of};
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernels::{CORE, rows, workgroups};
use dynamis_scene::Count;
use dynamis_scene::Frame;
use dynamis_scene::SceneStream;

pub struct Commands {
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
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            reset_counters: Stage::build(
                context,
                "reset_counters",
                workgroups(
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
                rows(
                    context,
                    include_str!("../shaders/body_move_gather.wgsl"),
                    CORE,
                    Count::BodyMoves.field(),
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
                rows(
                    context,
                    include_str!("../shaders/body_move_scatter.wgsl"),
                    CORE,
                    Count::BodyMoves.field(),
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
                rows(
                    context,
                    include_str!("../shaders/body_edits.wgsl"),
                    CORE,
                    Count::EditRuns.field(),
                ),
                streams,
                &[
                    ("edits", SceneStream::BodyEdits.whole()),
                    ("edit_runs", SceneStream::BodyEditRuns.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("wake_flags", SceneStream::WakeFlags.whole()),
                    ("params", SceneStream::Params.whole()),
                    ("slept_count", dynamis_scene::counter(COUNTER_SLEPT)),
                    ("woke_count", dynamis_scene::counter(COUNTER_WOKE)),
                ],
                &[],
            ),
            row_of_body: Stage::build(
                context,
                "row_of_body",
                rows(
                    context,
                    include_str!("../shaders/row_of_body.wgsl"),
                    CORE,
                    Count::BodyMoves.field(),
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
                rows(
                    context,
                    include_str!("../shaders/constraint_rows.wgsl"),
                    CORE,
                    Count::Constraints.field(),
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
                rows(
                    context,
                    include_str!("../shaders/constraint_move_gather.wgsl"),
                    CORE,
                    Count::ConstraintMoves.field(),
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
                rows(
                    context,
                    include_str!("../shaders/constraint_move_scatter.wgsl"),
                    CORE,
                    Count::ConstraintMoves.field(),
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
                rows(
                    context,
                    include_str!("../shaders/activity.wgsl"),
                    CORE,
                    Count::Bodies.field(),
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("body_activity", RigidStream::BodyActivity.whole()),
                    ("active_count", dynamis_scene::counter(COUNTER_ACTIVE)),
                ],
                &[],
            ),
            joint_filter: Stage::build(
                context,
                "joint_filter",
                rows(
                    context,
                    include_str!("../shaders/joint_filter.wgsl"),
                    CORE,
                    Count::Constraints.field(),
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
                    ("joint_count", dynamis_scene::counter(COUNTER_JOINTS)),
                    ("constraint_rows", RigidStream::ConstraintRows.whole()),
                ],
                &[],
            ),
        }
    }

    pub fn reset(&self, recorder: &mut ComputeRecorder, streams: &impl Resources) {
        self.reset_counters.record_workgroups(
            recorder,
            streams,
            workgroups_of(COUNTER_COUNT as u32),
        );
    }

    pub fn record(&self, recorder: &mut ComputeRecorder, streams: &impl Resources, frame: &Frame) {
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

    pub fn record_moves(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
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

    pub fn record_edits(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &Frame,
    ) {
        self.body_edits
            .record_rows(recorder, streams, Count::EditRuns.rows(&frame.params));
    }
}
