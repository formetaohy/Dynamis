use super::FrameParams;
use super::stage::{BLOCKS, CORE, POSITION_CONTACT, Stage, whole};
use crate::dynamics::buffers::WorldBuffers;
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
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        let segments = buffers.solver_segments.slot();
        let block_count = buffers.counter(COUNTER_BLOCKS);
        let reset = Stage::build(
            context,
            "solver reset",
            include_str!("shaders/solver_reset.wgsl"),
            per_row,
            CORE,
            &[
                ("params", whole(&buffers.params)),
                ("first_a", whole(&buffers.solver_first_a)),
                ("first_b", whole(&buffers.solver_first_b)),
                ("block_counts", whole(&buffers.solver_block_counts)),
                ("contact_counts", whole(&buffers.solver_contact_counts)),
                ("resolution", whole(&buffers.solver_resolution)),
                ("contributions", whole(&buffers.solver_contributions)),
            ],
            &[],
        );
        let total = Stage::build(
            context,
            "solver total",
            include_str!("shaders/solver_total.wgsl"),
            per_row,
            CORE,
            &[
                ("params", whole(&buffers.params)),
                ("contacts", whole(&buffers.contacts)),
                ("constraint_runtime", whole(&buffers.constraint_runtime)),
                ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                ("segments", segments),
                ("block_count", block_count),
            ],
            &[],
        );
        let blocks = Stage::build(
            context,
            "solver blocks",
            include_str!("shaders/solver_blocks.wgsl"),
            per_row,
            CORE,
            &[
                ("params", whole(&buffers.params)),
                ("body_states", whole(&buffers.body_states)),
                ("body_descs", whole(&buffers.body_descriptors)),
                ("contacts", whole(&buffers.contacts)),
                ("constraint_descs", whole(&buffers.constraint_descriptors)),
                ("segments", segments),
                ("block_first_body", whole(&buffers.solver_block_first_body)),
                (
                    "block_second_body",
                    whole(&buffers.solver_block_second_body),
                ),
                ("a_bodies", whole(&buffers.solver_a_bodies)),
                ("a_payload", whole(&buffers.solver_a_payload)),
                ("collider_owners", whole(&buffers.collider_owners)),
            ],
            &[],
        );
        let pair_order = Stage::build(
            context,
            "solver pair order",
            include_str!("shaders/solver_pair_order.wgsl"),
            per_row,
            CORE,
            &[
                ("a_payload", whole(&buffers.solver_a_payload)),
                (
                    "block_second_body",
                    whole(&buffers.solver_block_second_body),
                ),
                ("b_bodies", whole(&buffers.solver_b_bodies)),
                ("b_blocks", whole(&buffers.solver_b_blocks)),
                ("block_count", block_count),
            ],
            &[],
        );
        let boundaries = Stage::build(
            context,
            "solver boundaries",
            include_str!("shaders/solver_boundaries.wgsl"),
            per_row,
            CORE,
            &[
                ("segments", segments),
                ("a_bodies", whole(&buffers.solver_a_bodies)),
                ("b_bodies", whole(&buffers.solver_b_bodies)),
                ("a_payload", whole(&buffers.solver_a_payload)),
                ("first_a", whole(&buffers.solver_first_a)),
                ("first_b", whole(&buffers.solver_first_b)),
                ("contacts", whole(&buffers.contacts)),
                ("constraint_runtime", whole(&buffers.constraint_runtime)),
                ("block_counts", whole(&buffers.solver_block_counts)),
                ("contact_counts", whole(&buffers.solver_contact_counts)),
            ],
            &[],
        );
        let block_solve = Stage::build_warm(
            context,
            "solver block solve",
            include_str!("shaders/solver_block_solve.wgsl"),
            per_row,
            BLOCKS,
            &[
                ("params", whole(&buffers.params)),
                ("body_states", whole(&buffers.body_states)),
                ("body_descs", whole(&buffers.body_descriptors)),
                ("contacts", whole(&buffers.contacts)),
                ("constraint_descs", whole(&buffers.constraint_descriptors)),
                ("constraint_runtime", whole(&buffers.constraint_runtime)),
                ("wake_flags", whole(&buffers.wake_flags)),
                ("segments", segments),
                ("a_payload", whole(&buffers.solver_a_payload)),
                ("block_deltas", whole(&buffers.solver_block_deltas)),
                ("block_counts", whole(&buffers.solver_block_counts)),
                ("collider_owners", whole(&buffers.collider_owners)),
                ("block_count", block_count),
            ],
            &[],
        );
        let block_apply = Stage::build(
            context,
            "solver block apply",
            include_str!("shaders/solver_block_apply.wgsl"),
            per_row,
            CORE,
            &[
                ("params", whole(&buffers.params)),
                ("body_states", whole(&buffers.body_states)),
                ("first_a", whole(&buffers.solver_first_a)),
                ("first_b", whole(&buffers.solver_first_b)),
                ("a_bodies", whole(&buffers.solver_a_bodies)),
                ("b_bodies", whole(&buffers.solver_b_bodies)),
                ("b_blocks", whole(&buffers.solver_b_blocks)),
                ("block_counts", whole(&buffers.solver_block_counts)),
                ("block_deltas", whole(&buffers.solver_block_deltas)),
                ("block_count", block_count),
            ],
            &[],
        );
        let position_block = Stage::build(
            context,
            "position block",
            include_str!("shaders/position_block.wgsl"),
            per_row,
            POSITION_CONTACT,
            &[
                ("params", whole(&buffers.params)),
                ("body_states", whole(&buffers.body_states)),
                ("body_descs", whole(&buffers.body_descriptors)),
                ("contacts", whole(&buffers.contacts)),
                ("segments", segments),
                ("a_payload", whole(&buffers.solver_a_payload)),
                (
                    "block_corrections",
                    whole(&buffers.solver_block_corrections),
                ),
                ("resolution", whole(&buffers.solver_resolution)),
                ("collider_owners", whole(&buffers.collider_owners)),
                ("contributions", whole(&buffers.solver_contributions)),
                ("block_count", block_count),
            ],
            &[],
        );
        let position_apply = Stage::build(
            context,
            "position apply",
            include_str!("shaders/position_apply.wgsl"),
            per_row,
            CORE,
            &[
                ("params", whole(&buffers.params)),
                ("body_states", whole(&buffers.body_states)),
                ("first_a", whole(&buffers.solver_first_a)),
                ("first_b", whole(&buffers.solver_first_b)),
                ("a_bodies", whole(&buffers.solver_a_bodies)),
                ("b_bodies", whole(&buffers.solver_b_bodies)),
                ("b_blocks", whole(&buffers.solver_b_blocks)),
                ("contact_counts", whole(&buffers.solver_contact_counts)),
                (
                    "block_corrections",
                    whole(&buffers.solver_block_corrections),
                ),
                ("block_count", block_count),
                ("resolution", whole(&buffers.solver_resolution)),
                ("contributions", whole(&buffers.solver_contributions)),
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

    pub(super) fn record_prepare(&self, recorder: &mut ComputeRecorder, body_count: u32) {
        self.reset.record(recorder, body_count);
        self.total.record_workgroups(recorder, 1);
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        params: &FrameParams,
        sort: &RadixSort,
    ) {
        let blocks = buffers.block_capacity();
        let words = buffers.body_words();
        let block_count = buffers.counter(COUNTER_BLOCKS);
        self.blocks.record_stride(recorder, blocks);
        sort.sort(
            recorder,
            &buffers.sort_lanes(
                block_count,
                &buffers.solver_a_bodies,
                &buffers.solver_a_payload,
            ),
            words,
            0,
            blocks,
        );
        self.pair_order.record_stride(recorder, blocks);
        sort.sort(
            recorder,
            &buffers.sort_lanes(
                block_count,
                &buffers.solver_b_bodies,
                &buffers.solver_b_blocks,
            ),
            words,
            0,
            blocks,
        );
        self.boundaries.record_stride(recorder, blocks);
        self.block_solve.record_warm_stride(recorder, blocks);
        self.block_apply.record(recorder, params.dynamic_count);
        for _ in 0..params.solve_iterations {
            self.block_solve.record_stride(recorder, blocks);
            self.block_apply.record(recorder, params.dynamic_count);
        }
    }

    pub(super) fn record_position_iterations(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        params: &FrameParams,
    ) {
        let blocks = buffers.block_capacity();
        for _ in 0..params.position_iterations {
            self.position_block.record_stride(recorder, blocks);
            self.position_apply.record(recorder, params.dynamic_count);
        }
    }
}
