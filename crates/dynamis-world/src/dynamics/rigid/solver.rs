use super::Count;
use super::streams::RigidStream;
use crate::dynamics::Frame;
use crate::dynamics::engine::Stage;
use crate::dynamics::scene::SceneStream;
use crate::dynamics::shader;
use crate::dynamics::shader::{BLOCKS, CORE, POSITION_CORRECTION};
use crate::dynamics::streams::Streams;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_BLOCKS, COUNTER_CONTACTS};
use dynamis_sort::RadixSort;

pub(super) struct Solver {
    reset: Stage,
    total: Stage,
    blocks: Stage,
    pair_order: Stage,
    boundaries: Stage,
    block_solve: Stage,
    block_apply: Stage,
    position_block: Stage,
    position_apply: Stage,
}

impl Solver {
    pub(super) fn build(context: &GpuContext, streams: &Streams) -> Self {
        let segments = RigidStream::SolverSegments.whole();
        let block_count = streams.scene.counter(COUNTER_BLOCKS);
        let reset = Stage::build(
            context,
            "solver reset",
            shader::rows(
                context,
                include_str!("../shaders/solver_reset.wgsl"),
                CORE,
                Count::Bodies,
            ),
            streams,
            &[
                ("params", SceneStream::Params.whole()),
                ("first_a", RigidStream::SolverFirstA.whole()),
                ("first_b", RigidStream::SolverFirstB.whole()),
                ("block_counts", RigidStream::SolverBlockCounts.whole()),
                ("contact_counts", RigidStream::SolverContactCounts.whole()),
                ("resolution", RigidStream::SolverResolution.whole()),
                ("contributions", RigidStream::SolverContributions.whole()),
            ],
            &[],
        );
        let total = Stage::build(
            context,
            "solver total",
            shader::workgroups(context, include_str!("../shaders/solver_total.wgsl"), CORE),
            streams,
            &[
                ("params", SceneStream::Params.whole()),
                ("contacts", RigidStream::Contacts.whole()),
                ("constraint_runtime", SceneStream::ConstraintRuntime.whole()),
                ("contact_count", streams.scene.counter(COUNTER_CONTACTS)),
                ("segments", segments),
                ("block_count", block_count),
            ],
            &[],
        );
        let blocks = Stage::build(
            context,
            "solver blocks",
            shader::stream(
                context,
                include_str!("../shaders/solver_blocks.wgsl"),
                CORE,
                "work",
                RigidStream::SolverAPayload,
            ),
            streams,
            &[
                ("params", SceneStream::Params.whole()),
                ("body_states", SceneStream::BodyStates.whole()),
                ("body_descs", SceneStream::BodyDescriptors.whole()),
                ("contacts", RigidStream::Contacts.whole()),
                ("segments", segments),
                (
                    "block_first_body",
                    RigidStream::SolverBlockFirstBody.whole(),
                ),
                (
                    "block_second_body",
                    RigidStream::SolverBlockSecondBody.whole(),
                ),
                ("a_bodies", RigidStream::SolverABodies.whole()),
                ("a_payload", RigidStream::SolverAPayload.whole()),
                ("collider_owners", SceneStream::ColliderOwners.whole()),
                ("target_speeds", RigidStream::ContactTargetSpeeds.whole()),
                ("constraint_rows", RigidStream::ConstraintRows.whole()),
            ],
            &[],
        );
        let pair_order = Stage::build(
            context,
            "solver pair order",
            shader::stream(
                context,
                include_str!("../shaders/solver_pair_order.wgsl"),
                CORE,
                "work",
                RigidStream::SolverAPayload,
            ),
            streams,
            &[
                ("a_payload", RigidStream::SolverAPayload.whole()),
                (
                    "block_second_body",
                    RigidStream::SolverBlockSecondBody.whole(),
                ),
                ("b_bodies", RigidStream::SolverBBodies.whole()),
                ("b_blocks", RigidStream::SolverBBlocks.whole()),
                ("block_count", block_count),
            ],
            &[],
        );
        let boundaries = Stage::build(
            context,
            "solver boundaries",
            shader::stream(
                context,
                include_str!("../shaders/solver_boundaries.wgsl"),
                CORE,
                "work",
                RigidStream::SolverAPayload,
            ),
            streams,
            &[
                ("segments", segments),
                ("a_bodies", RigidStream::SolverABodies.whole()),
                ("b_bodies", RigidStream::SolverBBodies.whole()),
                ("a_payload", RigidStream::SolverAPayload.whole()),
                ("first_a", RigidStream::SolverFirstA.whole()),
                ("first_b", RigidStream::SolverFirstB.whole()),
                ("contacts", RigidStream::Contacts.whole()),
                ("constraint_runtime", SceneStream::ConstraintRuntime.whole()),
                ("block_counts", RigidStream::SolverBlockCounts.whole()),
                ("contact_counts", RigidStream::SolverContactCounts.whole()),
            ],
            &[],
        );
        let block_solve = Stage::build(
            context,
            "solver_block_solve",
            shader::stream_warm(
                context,
                include_str!("../shaders/solver_block_solve.wgsl"),
                BLOCKS,
                "work",
                RigidStream::SolverAPayload,
            ),
            streams,
            &[
                ("params", SceneStream::Params.whole()),
                ("body_states", SceneStream::BodyStates.whole()),
                ("body_descs", SceneStream::BodyDescriptors.whole()),
                ("contacts", RigidStream::Contacts.whole()),
                (
                    "constraint_descs",
                    SceneStream::ConstraintDescriptors.whole(),
                ),
                ("constraint_runtime", SceneStream::ConstraintRuntime.whole()),
                ("wake_flags", RigidStream::WakeFlags.whole()),
                ("segments", segments),
                ("a_payload", RigidStream::SolverAPayload.whole()),
                ("block_deltas", RigidStream::SolverBlockDeltas.whole()),
                ("block_counts", RigidStream::SolverBlockCounts.whole()),
                ("collider_owners", SceneStream::ColliderOwners.whole()),
                ("target_speeds", RigidStream::ContactTargetSpeeds.whole()),
                ("block_count", block_count),
                ("constraint_rows", RigidStream::ConstraintRows.whole()),
            ],
            &[],
        );
        let block_apply = Stage::build(
            context,
            "solver block apply",
            shader::rows(
                context,
                include_str!("../shaders/solver_block_apply.wgsl"),
                CORE,
                Count::Dynamic,
            ),
            streams,
            &[
                ("params", SceneStream::Params.whole()),
                ("body_states", SceneStream::BodyStates.whole()),
                ("first_a", RigidStream::SolverFirstA.whole()),
                ("first_b", RigidStream::SolverFirstB.whole()),
                ("a_bodies", RigidStream::SolverABodies.whole()),
                ("b_bodies", RigidStream::SolverBBodies.whole()),
                ("b_blocks", RigidStream::SolverBBlocks.whole()),
                ("block_counts", RigidStream::SolverBlockCounts.whole()),
                ("block_deltas", RigidStream::SolverBlockDeltas.whole()),
                ("block_count", block_count),
            ],
            &[],
        );
        let position_block = Stage::build(
            context,
            "position block",
            shader::stream(
                context,
                include_str!("../shaders/position_block.wgsl"),
                POSITION_CORRECTION,
                "work",
                RigidStream::SolverAPayload,
            ),
            streams,
            &[
                ("params", SceneStream::Params.whole()),
                ("body_states", SceneStream::BodyStates.whole()),
                ("body_descs", SceneStream::BodyDescriptors.whole()),
                ("contacts", RigidStream::Contacts.whole()),
                (
                    "constraint_descs",
                    SceneStream::ConstraintDescriptors.whole(),
                ),
                ("constraint_runtime", SceneStream::ConstraintRuntime.whole()),
                ("segments", segments),
                ("a_payload", RigidStream::SolverAPayload.whole()),
                (
                    "block_corrections",
                    RigidStream::SolverBlockCorrections.whole(),
                ),
                ("resolution", RigidStream::SolverResolution.whole()),
                ("collider_owners", SceneStream::ColliderOwners.whole()),
                ("contributions", RigidStream::SolverContributions.whole()),
                ("block_count", block_count),
                ("constraint_rows", RigidStream::ConstraintRows.whole()),
            ],
            &[],
        );
        let position_apply = Stage::build(
            context,
            "position apply",
            shader::rows(
                context,
                include_str!("../shaders/position_apply.wgsl"),
                CORE,
                Count::Dynamic,
            ),
            streams,
            &[
                ("params", SceneStream::Params.whole()),
                ("body_states", SceneStream::BodyStates.whole()),
                ("first_a", RigidStream::SolverFirstA.whole()),
                ("first_b", RigidStream::SolverFirstB.whole()),
                ("a_bodies", RigidStream::SolverABodies.whole()),
                ("b_bodies", RigidStream::SolverBBodies.whole()),
                ("b_blocks", RigidStream::SolverBBlocks.whole()),
                (
                    "block_corrections",
                    RigidStream::SolverBlockCorrections.whole(),
                ),
                ("block_count", block_count),
                ("resolution", RigidStream::SolverResolution.whole()),
                ("contributions", RigidStream::SolverContributions.whole()),
            ],
            &[],
        );
        Self {
            reset,
            total,
            blocks,
            pair_order,
            boundaries,
            block_solve,
            block_apply,
            position_block,
            position_apply,
        }
    }

    pub(super) fn record_prepare(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &Streams,
        frame: &Frame,
    ) {
        self.reset
            .record_rows(recorder, streams, Count::Bodies.rows(&frame.params));
        self.total.record_workgroups(recorder, streams, 1);
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &Streams,
        frame: &Frame,
        sort: &RadixSort,
    ) {
        let blocks = streams.rigid.block_capacity();
        let words = streams.scene.body_words();
        let block_count = streams.scene.counter(COUNTER_BLOCKS);
        self.blocks.record_stream(recorder, streams);
        sort.sort(
            recorder,
            &streams.sort_lanes(
                block_count,
                RigidStream::SolverABodies.whole(),
                RigidStream::SolverAPayload.whole(),
            ),
            words,
            0,
            blocks,
        );
        self.pair_order.record_stream(recorder, streams);
        sort.sort(
            recorder,
            &streams.sort_lanes(
                block_count,
                RigidStream::SolverBBodies.whole(),
                RigidStream::SolverBBlocks.whole(),
            ),
            words,
            0,
            blocks,
        );
        self.boundaries.record_stream(recorder, streams);
        self.block_solve.record_warm(recorder, streams);
        self.block_apply
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        for _ in 0..frame.params.solve_iterations {
            self.block_solve.record_stream(recorder, streams);
            self.block_apply
                .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        }
    }

    pub(super) fn record_position_iterations(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &Streams,
        frame: &Frame,
    ) {
        for _ in 0..frame.params.position_iterations {
            self.position_block.record_stream(recorder, streams);
            self.position_apply
                .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        }
    }
}
