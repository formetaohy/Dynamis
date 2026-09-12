use super::Count;
use super::Frame;
use super::buffers::RigidBuffers;
use super::shader;
use super::shader::CORE;
use crate::dynamics::engine::{Stage, whole, workgroups_of};
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
                &[("counters", whole(&buffers.counters))],
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
                shader::rows(
                    context,
                    include_str!("shaders/body_move_scatter.wgsl"),
                    CORE,
                    Count::BodyMoves,
                ),
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
                shader::rows(
                    context,
                    include_str!("shaders/body_edits.wgsl"),
                    CORE,
                    Count::EditRuns,
                ),
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
                shader::rows(
                    context,
                    include_str!("shaders/row_of_body.wgsl"),
                    CORE,
                    Count::BodyMoves,
                ),
                &[
                    ("body_states", whole(&buffers.body_states)),
                    ("row_moves", whole(&buffers.body_row_moves)),
                    ("row_of_body", whole(&buffers.body_row_of_id)),
                    ("params", whole(&buffers.params)),
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
                &[
                    ("params", whole(&buffers.params)),
                    ("constraint_descs", whole(&buffers.constraint_descriptors)),
                    ("row_of_body", whole(&buffers.body_row_of_id)),
                    ("constraint_rows", whole(&buffers.constraint_rows)),
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
                shader::rows(
                    context,
                    include_str!("shaders/constraint_move_scatter.wgsl"),
                    CORE,
                    Count::ConstraintMoves,
                ),
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
                shader::rows(
                    context,
                    include_str!("shaders/activity.wgsl"),
                    CORE,
                    Count::Bodies,
                ),
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
                shader::rows(
                    context,
                    include_str!("shaders/joint_filter.wgsl"),
                    CORE,
                    Count::Constraints,
                ),
                &[
                    ("params", whole(&buffers.params)),
                    ("constraint_descs", whole(&buffers.constraint_descriptors)),
                    ("constraint_runtime", whole(&buffers.constraint_runtime)),
                    ("joint_major", whole(&buffers.joint_filter_major)),
                    ("joint_minor", whole(&buffers.joint_filter_minor)),
                    ("joint_count", buffers.counter(COUNTER_JOINTS)),
                    ("constraint_rows", whole(&buffers.constraint_rows)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn reset(&self, recorder: &mut ComputeRecorder) {
        self.reset_counters
            .record_workgroups(recorder, workgroups_of(COUNTER_COUNT as u32));
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, frame: &Frame) {
        self.reset(recorder);
        self.record_moves(recorder, frame);
        self.record_edits(recorder, frame);
        self.constraint_rows
            .record_rows(recorder, Count::Constraints.rows(&frame.params));
        self.joint_filter
            .record_rows(recorder, Count::Constraints.rows(&frame.params));
        self.activity
            .record_rows(recorder, Count::Bodies.rows(&frame.params));
    }

    pub(super) fn record_moves(&self, recorder: &mut ComputeRecorder, frame: &Frame) {
        self.body_move_gather
            .record_rows(recorder, Count::BodyMoves.rows(&frame.params));
        self.body_move_scatter
            .record_rows(recorder, Count::BodyMoves.rows(&frame.params));
        self.row_of_body
            .record_rows(recorder, Count::BodyMoves.rows(&frame.params));
        self.constraint_move_gather
            .record_rows(recorder, Count::ConstraintMoves.rows(&frame.params));
        self.constraint_move_scatter
            .record_rows(recorder, Count::ConstraintMoves.rows(&frame.params));
    }

    pub(super) fn record_edits(&self, recorder: &mut ComputeRecorder, frame: &Frame) {
        self.body_edits
            .record_rows(recorder, Count::EditRuns.rows(&frame.params));
    }
}
