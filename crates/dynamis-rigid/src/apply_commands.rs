use super::streams::RigidStream;
use crate::RigidFrame;
use dynamis_abi::Count;
use dynamis_abi::{
    COUNTER_ACTIVE, COUNTER_JOINTS, COUNTER_SLEPT, COUNTER_STEP_RESET_SLOTS, COUNTER_WOKE,
};
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::ResourceSource;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{PassRuntime, Stage};
use dynamis_shader::workgroups_of;
use dynamis_shader::{CORE, rows, workgroups};
use dynamis_state::StateStream;

pub struct ApplyCommands {
    reset_counters: Stage,
    clear_inputs: Stage,
    body_move_gather: Stage,
    body_move_scatter: Stage,
    body_edits: Stage,
    row_of_body: Stage,
    constraint_rows: Stage,
    constraint_move_gather: Stage,
    constraint_move_scatter: Stage,
    row_of_constraint: Stage,
    joint_filter: Stage,
    activity: Stage,
}

impl PassRuntime<RigidFrame> for ApplyCommands {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        let reset = dynamis_abi::step_reset_wgsl();
        let reset_fragments = [reset.as_str()];
        Self {
            reset_counters: Stage::build(
                context,
                "reset_counters",
                workgroups(
                    context,
                    include_str!("../shaders/reset_counters.wgsl"),
                    &reset_fragments,
                ),
                streams,
                &[("counters", StateStream::Counters.whole())],
                &[],
            ),
            clear_inputs: Stage::build(
                context,
                "clear_inputs",
                rows(
                    context,
                    include_str!("../shaders/clear_inputs.wgsl"),
                    CORE,
                    Count::Bodies.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                ],
                &[],
            ),
            body_move_gather: Stage::build(
                context,
                "body_move_gather",
                rows(
                    context,
                    include_str!("../shaders/body_move_gather.wgsl"),
                    dynamis_shader::COUNTERS,
                    Count::BodyMoves.bound(),
                ),
                streams,
                &[
                    ("body_states", StateStream::BodyStates.whole()),
                    ("state_scratch", RigidStream::BodyStateScratch.whole()),
                    ("row_moves", StateStream::BodyRowMoves.whole()),
                    ("fresh_rows", StateStream::BodyFreshRows.whole()),
                    ("row_streams", StateStream::RowStreams.whole()),
                    ("counters", StateStream::Counters.whole()),
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
                    Count::BodyMoves.bound(),
                ),
                streams,
                &[
                    ("body_states", StateStream::BodyStates.whole()),
                    ("state_scratch", RigidStream::BodyStateScratch.whole()),
                    ("row_moves", StateStream::BodyRowMoves.whole()),
                    ("row_streams", StateStream::RowStreams.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
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
                    Count::BodyEditRuns.bound(),
                ),
                streams,
                &[
                    ("edits", StateStream::BodyEdits.whole()),
                    ("edit_runs", StateStream::BodyEditRuns.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                    ("row_streams", StateStream::RowStreams.whole()),
                    ("slept_count", dynamis_state::counter(COUNTER_SLEPT)),
                    ("woke_count", dynamis_state::counter(COUNTER_WOKE)),
                    ("body_admitted", BroadphaseStream::BodyAdmitted.whole()),
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
                    Count::BodyMoves.bound(),
                ),
                streams,
                &[
                    ("body_states", StateStream::BodyStates.whole()),
                    ("row_moves", StateStream::BodyRowMoves.whole()),
                    ("row_of_body", StateStream::BodyRowOfId.whole()),
                    ("row_streams", StateStream::RowStreams.whole()),
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
                    Count::Constraints.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    (
                        "constraint_descs",
                        StateStream::ConstraintDescriptors.whole(),
                    ),
                    ("row_of_body", StateStream::BodyRowOfId.whole()),
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
                    dynamis_shader::COUNTERS,
                    Count::ConstraintMoves.bound(),
                ),
                streams,
                &[
                    ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                    ("constraint_scratch", RigidStream::ConstraintScratch.whole()),
                    ("row_moves", StateStream::ConstraintRowMoves.whole()),
                    ("fresh_rows", StateStream::ConstraintFreshRows.whole()),
                    ("row_streams", StateStream::RowStreams.whole()),
                    (
                        "constraint_descs",
                        StateStream::ConstraintDescriptors.whole(),
                    ),
                    ("row_of_body", StateStream::BodyRowOfId.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("counters", StateStream::Counters.whole()),
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
                    Count::ConstraintMoves.bound(),
                ),
                streams,
                &[
                    ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                    ("constraint_scratch", RigidStream::ConstraintScratch.whole()),
                    ("row_moves", StateStream::ConstraintRowMoves.whole()),
                    ("row_streams", StateStream::RowStreams.whole()),
                ],
                &[],
            ),
            row_of_constraint: Stage::build(
                context,
                "row_of_constraint",
                rows(
                    context,
                    include_str!("../shaders/row_of_constraint.wgsl"),
                    CORE,
                    Count::ConstraintMoves.bound(),
                ),
                streams,
                &[
                    ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                    ("row_moves", StateStream::ConstraintRowMoves.whole()),
                    ("row_of_constraint", StateStream::ConstraintRowOfId.whole()),
                    ("row_streams", StateStream::RowStreams.whole()),
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
                    Count::Bodies.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("body_activity", RigidStream::BodyActivity.whole()),
                    ("body_motion", RigidStream::BodyMotion.whole()),
                    ("active_count", dynamis_state::counter(COUNTER_ACTIVE)),
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
                    Count::Constraints.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    (
                        "constraint_descs",
                        StateStream::ConstraintDescriptors.whole(),
                    ),
                    ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                    ("joint_major", RigidStream::JointFilterMajor.whole()),
                    ("joint_minor", RigidStream::JointFilterMinor.whole()),
                    ("joint_count", dynamis_state::counter(COUNTER_JOINTS)),
                    ("constraint_rows", RigidStream::ConstraintRows.whole()),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        self.reset(recorder, streams);
        self.clear_inputs.record_rows(
            recorder,
            streams,
            Count::Bodies.rows(&frame.params, &frame.rows),
        );
        self.record_moves(recorder, streams, frame);
        self.record_edits(recorder, streams, frame);
        self.constraint_rows.record_rows(
            recorder,
            streams,
            Count::Constraints.rows(&frame.params, &frame.rows),
        );
        self.joint_filter.record_rows(
            recorder,
            streams,
            Count::Constraints.rows(&frame.params, &frame.rows),
        );
        self.activity.record_rows(
            recorder,
            streams,
            Count::Bodies.rows(&frame.params, &frame.rows),
        );
    }
}

impl ApplyCommands {
    fn reset(&mut self, recorder: &mut ComputeRecorder<'_>, streams: &impl ResourceSource) {
        self.reset_counters.record_workgroups(
            recorder,
            streams,
            workgroups_of(COUNTER_STEP_RESET_SLOTS.len() as u32),
        );
    }

    fn record_moves(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        self.body_move_gather.record_rows(
            recorder,
            streams,
            Count::BodyMoves.rows(&frame.params, &frame.rows),
        );
        self.body_move_scatter.record_rows(
            recorder,
            streams,
            Count::BodyMoves.rows(&frame.params, &frame.rows),
        );
        self.row_of_body.record_rows(
            recorder,
            streams,
            Count::BodyMoves.rows(&frame.params, &frame.rows),
        );
        self.constraint_move_gather.record_rows(
            recorder,
            streams,
            Count::ConstraintMoves.rows(&frame.params, &frame.rows),
        );
        self.constraint_move_scatter.record_rows(
            recorder,
            streams,
            Count::ConstraintMoves.rows(&frame.params, &frame.rows),
        );
        self.row_of_constraint.record_rows(
            recorder,
            streams,
            Count::ConstraintMoves.rows(&frame.params, &frame.rows),
        );
    }

    fn record_edits(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        self.body_edits.record_rows(
            recorder,
            streams,
            Count::BodyEditRuns.rows(&frame.params, &frame.rows),
        );
    }
}
