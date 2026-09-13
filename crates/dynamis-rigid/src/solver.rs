use super::streams::RigidStream;
use crate::RigidFrame;
use crate::sort;
use dynamis_abi::Count;
use dynamis_abi::{COUNTER_BLOCKS, COUNTER_CONTACTS};
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernel::{CORE, rows, stream, stream_warm, workgroups};
use dynamis_pass::Resources;
use dynamis_pass::Stage;
use dynamis_sort::RadixSort;
use dynamis_state::StateStream;

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
        let block_count = dynamis_state::counter(COUNTER_BLOCKS);
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
                ("params", StateStream::Params.whole()),
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
                ("params", StateStream::Params.whole()),
                ("contacts", RigidStream::Contacts.whole()),
                ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
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
                ("params", StateStream::Params.whole()),
                ("body_states", StateStream::BodyStates.whole()),
                ("body_descs", StateStream::BodyDescriptors.whole()),
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
                ("collider_owners", StateStream::ColliderOwners.whole()),
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
                ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
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
                ("params", StateStream::Params.whole()),
                ("body_states", StateStream::BodyStates.whole()),
                ("body_descs", StateStream::BodyDescriptors.whole()),
                ("contacts", RigidStream::Contacts.whole()),
                (
                    "constraint_descs",
                    StateStream::ConstraintDescriptors.whole(),
                ),
                ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                ("segments", segments),
                ("a_payload", RigidStream::SolverAPayload.whole()),
                ("block_deltas", RigidStream::SolverBlockDeltas.whole()),
                ("block_counts", RigidStream::SolverBlockCounts.whole()),
                ("collider_owners", StateStream::ColliderOwners.whole()),
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
                ("params", StateStream::Params.whole()),
                ("body_states", StateStream::BodyStates.whole()),
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
                ("params", StateStream::Params.whole()),
                ("body_states", StateStream::BodyStates.whole()),
                ("body_descs", StateStream::BodyDescriptors.whole()),
                ("contacts", RigidStream::Contacts.whole()),
                (
                    "constraint_descs",
                    StateStream::ConstraintDescriptors.whole(),
                ),
                ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                ("segments", segments),
                ("a_payload", RigidStream::SolverAPayload.whole()),
                (
                    "block_corrections",
                    RigidStream::SolverBlockCorrections.whole(),
                ),
                ("resolution", RigidStream::SolverResolution.whole()),
                ("collider_owners", StateStream::ColliderOwners.whole()),
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
                ("params", StateStream::Params.whole()),
                ("body_states", StateStream::BodyStates.whole()),
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
        frame: &RigidFrame,
    ) {
        self.reset
            .record_rows(recorder, streams, Count::Bodies.rows(&frame.params));
        self.total.record_workgroups(recorder, streams, 1);
    }

    pub fn record_topology(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
        sort: &RadixSort,
    ) {
        let words = frame.shape.body_words;
        let block_count = dynamis_state::counter(COUNTER_BLOCKS);
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
        );
        self.boundaries.record_stream(recorder, streams);
    }

    pub fn record_warm(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.block_solve.record_warm(recorder, streams);
        self.block_apply
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
    }

    pub fn record_iterations(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        let dynamic = Count::Dynamic.rows(&frame.params);
        for _ in 0..frame.params.solve_iterations {
            self.block_solve.record_stream(recorder, streams);
            self.block_apply.record_rows(recorder, streams, dynamic);
        }
    }

    pub fn record_position_iterations(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        for _ in 0..frame.params.position_iterations {
            self.position_block.record_stream(recorder, streams);
            self.position_apply
                .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        }
    }
}
