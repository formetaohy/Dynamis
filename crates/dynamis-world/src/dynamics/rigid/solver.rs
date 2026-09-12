use super::Count;
use super::Frame;
use super::buffers::{RigidBuffers, StreamId};
use super::shader;
use super::shader::{BLOCKS, CORE, POSITION_CORRECTION};
use crate::dynamics::engine::Stage;
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
    pub(super) fn build(context: &GpuContext, buffers: &RigidBuffers) -> Self {
        let segments = StreamId::SolverSegments.whole();
        let block_count = buffers.counter(COUNTER_BLOCKS);
        let reset = Stage::build(
            context,
            "solver reset",
            shader::rows(
                context,
                include_str!("shaders/solver_reset.wgsl"),
                CORE,
                Count::Bodies,
            ),
            buffers,
            &[
                ("params", StreamId::Params.whole()),
                ("first_a", StreamId::SolverFirstA.whole()),
                ("first_b", StreamId::SolverFirstB.whole()),
                ("block_counts", StreamId::SolverBlockCounts.whole()),
                ("contact_counts", StreamId::SolverContactCounts.whole()),
                ("resolution", StreamId::SolverResolution.whole()),
                ("contributions", StreamId::SolverContributions.whole()),
            ],
            &[],
        );
        let total = Stage::build(
            context,
            "solver total",
            shader::workgroups(context, include_str!("shaders/solver_total.wgsl"), CORE),
            buffers,
            &[
                ("params", StreamId::Params.whole()),
                ("contacts", StreamId::Contacts.whole()),
                ("constraint_runtime", StreamId::ConstraintRuntime.whole()),
                ("contact_count", buffers.counter(COUNTER_CONTACTS)),
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
                include_str!("shaders/solver_blocks.wgsl"),
                CORE,
                "work",
                StreamId::SolverAPayload,
            ),
            buffers,
            &[
                ("params", StreamId::Params.whole()),
                ("body_states", StreamId::BodyStates.whole()),
                ("body_descs", StreamId::BodyDescriptors.whole()),
                ("contacts", StreamId::Contacts.whole()),
                ("segments", segments),
                ("block_first_body", StreamId::SolverBlockFirstBody.whole()),
                ("block_second_body", StreamId::SolverBlockSecondBody.whole()),
                ("a_bodies", StreamId::SolverABodies.whole()),
                ("a_payload", StreamId::SolverAPayload.whole()),
                ("collider_owners", StreamId::ColliderOwners.whole()),
                ("target_speeds", StreamId::ContactTargetSpeeds.whole()),
                ("constraint_rows", StreamId::ConstraintRows.whole()),
            ],
            &[],
        );
        let pair_order = Stage::build(
            context,
            "solver pair order",
            shader::stream(
                context,
                include_str!("shaders/solver_pair_order.wgsl"),
                CORE,
                "work",
                StreamId::SolverAPayload,
            ),
            buffers,
            &[
                ("a_payload", StreamId::SolverAPayload.whole()),
                ("block_second_body", StreamId::SolverBlockSecondBody.whole()),
                ("b_bodies", StreamId::SolverBBodies.whole()),
                ("b_blocks", StreamId::SolverBBlocks.whole()),
                ("block_count", block_count),
            ],
            &[],
        );
        let boundaries = Stage::build(
            context,
            "solver boundaries",
            shader::stream(
                context,
                include_str!("shaders/solver_boundaries.wgsl"),
                CORE,
                "work",
                StreamId::SolverAPayload,
            ),
            buffers,
            &[
                ("segments", segments),
                ("a_bodies", StreamId::SolverABodies.whole()),
                ("b_bodies", StreamId::SolverBBodies.whole()),
                ("a_payload", StreamId::SolverAPayload.whole()),
                ("first_a", StreamId::SolverFirstA.whole()),
                ("first_b", StreamId::SolverFirstB.whole()),
                ("contacts", StreamId::Contacts.whole()),
                ("constraint_runtime", StreamId::ConstraintRuntime.whole()),
                ("block_counts", StreamId::SolverBlockCounts.whole()),
                ("contact_counts", StreamId::SolverContactCounts.whole()),
            ],
            &[],
        );
        let block_solve = Stage::build(
            context,
            "solver_block_solve",
            shader::stream_warm(
                context,
                include_str!("shaders/solver_block_solve.wgsl"),
                BLOCKS,
                "work",
                StreamId::SolverAPayload,
            ),
            buffers,
            &[
                ("params", StreamId::Params.whole()),
                ("body_states", StreamId::BodyStates.whole()),
                ("body_descs", StreamId::BodyDescriptors.whole()),
                ("contacts", StreamId::Contacts.whole()),
                ("constraint_descs", StreamId::ConstraintDescriptors.whole()),
                ("constraint_runtime", StreamId::ConstraintRuntime.whole()),
                ("wake_flags", StreamId::WakeFlags.whole()),
                ("segments", segments),
                ("a_payload", StreamId::SolverAPayload.whole()),
                ("block_deltas", StreamId::SolverBlockDeltas.whole()),
                ("block_counts", StreamId::SolverBlockCounts.whole()),
                ("collider_owners", StreamId::ColliderOwners.whole()),
                ("target_speeds", StreamId::ContactTargetSpeeds.whole()),
                ("block_count", block_count),
                ("constraint_rows", StreamId::ConstraintRows.whole()),
            ],
            &[],
        );
        let block_apply = Stage::build(
            context,
            "solver block apply",
            shader::rows(
                context,
                include_str!("shaders/solver_block_apply.wgsl"),
                CORE,
                Count::Dynamic,
            ),
            buffers,
            &[
                ("params", StreamId::Params.whole()),
                ("body_states", StreamId::BodyStates.whole()),
                ("first_a", StreamId::SolverFirstA.whole()),
                ("first_b", StreamId::SolverFirstB.whole()),
                ("a_bodies", StreamId::SolverABodies.whole()),
                ("b_bodies", StreamId::SolverBBodies.whole()),
                ("b_blocks", StreamId::SolverBBlocks.whole()),
                ("block_counts", StreamId::SolverBlockCounts.whole()),
                ("block_deltas", StreamId::SolverBlockDeltas.whole()),
                ("block_count", block_count),
            ],
            &[],
        );
        let position_block = Stage::build(
            context,
            "position block",
            shader::stream(
                context,
                include_str!("shaders/position_block.wgsl"),
                POSITION_CORRECTION,
                "work",
                StreamId::SolverAPayload,
            ),
            buffers,
            &[
                ("params", StreamId::Params.whole()),
                ("body_states", StreamId::BodyStates.whole()),
                ("body_descs", StreamId::BodyDescriptors.whole()),
                ("contacts", StreamId::Contacts.whole()),
                ("constraint_descs", StreamId::ConstraintDescriptors.whole()),
                ("constraint_runtime", StreamId::ConstraintRuntime.whole()),
                ("segments", segments),
                ("a_payload", StreamId::SolverAPayload.whole()),
                (
                    "block_corrections",
                    StreamId::SolverBlockCorrections.whole(),
                ),
                ("resolution", StreamId::SolverResolution.whole()),
                ("collider_owners", StreamId::ColliderOwners.whole()),
                ("contributions", StreamId::SolverContributions.whole()),
                ("block_count", block_count),
                ("constraint_rows", StreamId::ConstraintRows.whole()),
            ],
            &[],
        );
        let position_apply = Stage::build(
            context,
            "position apply",
            shader::rows(
                context,
                include_str!("shaders/position_apply.wgsl"),
                CORE,
                Count::Dynamic,
            ),
            buffers,
            &[
                ("params", StreamId::Params.whole()),
                ("body_states", StreamId::BodyStates.whole()),
                ("first_a", StreamId::SolverFirstA.whole()),
                ("first_b", StreamId::SolverFirstB.whole()),
                ("a_bodies", StreamId::SolverABodies.whole()),
                ("b_bodies", StreamId::SolverBBodies.whole()),
                ("b_blocks", StreamId::SolverBBlocks.whole()),
                (
                    "block_corrections",
                    StreamId::SolverBlockCorrections.whole(),
                ),
                ("block_count", block_count),
                ("resolution", StreamId::SolverResolution.whole()),
                ("contributions", StreamId::SolverContributions.whole()),
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
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        self.reset
            .record_rows(recorder, buffers, Count::Bodies.rows(&frame.params));
        self.total.record_workgroups(recorder, buffers, 1);
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        frame: &Frame,
        sort: &RadixSort,
    ) {
        let blocks = buffers.block_capacity();
        let words = buffers.body_words();
        let block_count = buffers.counter(COUNTER_BLOCKS);
        self.blocks.record_stream(recorder, buffers);
        sort.sort(
            recorder,
            &buffers.sort_lanes(
                block_count,
                StreamId::SolverABodies.whole(),
                StreamId::SolverAPayload.whole(),
            ),
            words,
            0,
            blocks,
        );
        self.pair_order.record_stream(recorder, buffers);
        sort.sort(
            recorder,
            &buffers.sort_lanes(
                block_count,
                StreamId::SolverBBodies.whole(),
                StreamId::SolverBBlocks.whole(),
            ),
            words,
            0,
            blocks,
        );
        self.boundaries.record_stream(recorder, buffers);
        self.block_solve.record_warm(recorder, buffers);
        self.block_apply
            .record_rows(recorder, buffers, Count::Dynamic.rows(&frame.params));
        for _ in 0..frame.params.solve_iterations {
            self.block_solve.record_stream(recorder, buffers);
            self.block_apply
                .record_rows(recorder, buffers, Count::Dynamic.rows(&frame.params));
        }
    }

    pub(super) fn record_position_iterations(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        for _ in 0..frame.params.position_iterations {
            self.position_block.record_stream(recorder, buffers);
            self.position_apply
                .record_rows(recorder, buffers, Count::Dynamic.rows(&frame.params));
        }
    }
}
