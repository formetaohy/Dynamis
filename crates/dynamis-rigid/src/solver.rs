use super::streams::RigidStream;
use crate::sort;
use crate::streams::block_capacity;
use dynamis_abi::{COUNTER_BLOCKS, COUNTER_CONTACTS};
use dynamis_engine::Resources;
use dynamis_engine::Stage;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernels::{CORE, rows, stream, stream_warm, workgroups};
use dynamis_scene::Count;
use dynamis_scene::Frame;
use dynamis_scene::SceneStream;
use dynamis_sort::RadixSort;

const BLOCKS: &[&str] = &[
    include_str!("../shaders/solver_contact_block.wgsl"),
    include_str!("../shaders/solver_constraint_block.wgsl"),
];
const POSITION_CORRECTION: &[&str] = &[include_str!("../shaders/position_correction.wgsl")];

pub struct Solver {
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
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        let segments = RigidStream::SolverSegments.whole();
        let block_count = dynamis_scene::counter(COUNTER_BLOCKS);
        let reset = Stage::build(
            context,
            "solver reset",
            rows(
                context,
                include_str!("../shaders/solver_reset.wgsl"),
                CORE,
                Count::Bodies.field(),
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
            workgroups(context, include_str!("../shaders/solver_total.wgsl"), CORE),
            streams,
            &[
                ("params", SceneStream::Params.whole()),
                ("contacts", RigidStream::Contacts.whole()),
                ("constraint_runtime", SceneStream::ConstraintRuntime.whole()),
                ("contact_count", dynamis_scene::counter(COUNTER_CONTACTS)),
                ("segments", segments),
                ("block_count", block_count),
            ],
            &[],
        );
        let blocks = Stage::build(
            context,
            "solver blocks",
            stream(
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
            stream(
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
            stream(
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
            stream_warm(
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
                ("wake_flags", SceneStream::WakeFlags.whole()),
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
            rows(
                context,
                include_str!("../shaders/solver_block_apply.wgsl"),
                CORE,
                Count::Dynamic.field(),
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
            stream(
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
            rows(
                context,
                include_str!("../shaders/position_apply.wgsl"),
                CORE,
                Count::Dynamic.field(),
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

    pub fn record_prepare(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &Frame,
    ) {
        self.reset
            .record_rows(recorder, streams, Count::Bodies.rows(&frame.params));
        self.total.record_workgroups(recorder, streams, 1);
    }

    pub fn record(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &Frame,
        sort: &RadixSort,
    ) {
        let blocks = block_capacity(streams);
        let words = dynamis_scene::body_words(streams);
        let block_count = dynamis_scene::counter(COUNTER_BLOCKS);
        self.blocks.record_stream(recorder, streams);
        sort.sort(
            recorder,
            &sort::lanes(
                streams,
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
            &sort::lanes(
                streams,
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

    pub fn record_position_iterations(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &Frame,
    ) {
        for _ in 0..frame.params.position_iterations {
            self.position_block.record_stream(recorder, streams);
            self.position_apply
                .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        }
    }
}
