use super::Count;
use super::Frame;
use super::buffers::RigidBuffers;
use super::shader;
use super::shader::{BLOCKS, CORE, POSITION_CORRECTION};
use crate::dynamics::engine::{Stage, whole};
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
        let segments = buffers.solver_segments.slot();
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
            shader::workgroups(context, include_str!("shaders/solver_total.wgsl"), CORE),
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
            shader::stream(
                context,
                include_str!("shaders/solver_blocks.wgsl"),
                CORE,
                "work",
                buffers.block_capacity(),
            ),
            &[
                ("params", whole(&buffers.params)),
                ("body_states", whole(&buffers.body_states)),
                ("body_descs", whole(&buffers.body_descriptors)),
                ("contacts", whole(&buffers.contacts)),
                ("segments", segments),
                ("block_first_body", whole(&buffers.solver_block_first_body)),
                (
                    "block_second_body",
                    whole(&buffers.solver_block_second_body),
                ),
                ("a_bodies", whole(&buffers.solver_a_bodies)),
                ("a_payload", whole(&buffers.solver_a_payload)),
                ("collider_owners", whole(&buffers.collider_owners)),
                ("target_speeds", whole(&buffers.contact_target_speeds)),
                ("constraint_rows", whole(&buffers.constraint_rows)),
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
                buffers.block_capacity(),
            ),
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
            shader::stream(
                context,
                include_str!("shaders/solver_boundaries.wgsl"),
                CORE,
                "work",
                buffers.block_capacity(),
            ),
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
        let block_solve = Stage::build(
            context,
            "solver_block_solve",
            shader::stream_warm(
                context,
                include_str!("shaders/solver_block_solve.wgsl"),
                BLOCKS,
                "work",
                buffers.block_capacity(),
            ),
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
                ("target_speeds", whole(&buffers.contact_target_speeds)),
                ("block_count", block_count),
                ("constraint_rows", whole(&buffers.constraint_rows)),
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
            shader::stream(
                context,
                include_str!("shaders/position_block.wgsl"),
                POSITION_CORRECTION,
                "work",
                buffers.block_capacity(),
            ),
            &[
                ("params", whole(&buffers.params)),
                ("body_states", whole(&buffers.body_states)),
                ("body_descs", whole(&buffers.body_descriptors)),
                ("contacts", whole(&buffers.contacts)),
                ("constraint_descs", whole(&buffers.constraint_descriptors)),
                ("constraint_runtime", whole(&buffers.constraint_runtime)),
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
                ("constraint_rows", whole(&buffers.constraint_rows)),
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
            &[
                ("params", whole(&buffers.params)),
                ("body_states", whole(&buffers.body_states)),
                ("first_a", whole(&buffers.solver_first_a)),
                ("first_b", whole(&buffers.solver_first_b)),
                ("a_bodies", whole(&buffers.solver_a_bodies)),
                ("b_bodies", whole(&buffers.solver_b_bodies)),
                ("b_blocks", whole(&buffers.solver_b_blocks)),
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

    pub(super) fn record_prepare(&self, recorder: &mut ComputeRecorder, frame: &Frame) {
        self.reset
            .record_rows(recorder, Count::Bodies.rows(&frame.params));
        self.total.record_workgroups(recorder, 1);
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
        self.blocks.record_stream(recorder);
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
        self.pair_order.record_stream(recorder);
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
        self.boundaries.record_stream(recorder);
        self.block_solve.record_warm(recorder);
        self.block_apply
            .record_rows(recorder, Count::Dynamic.rows(&frame.params));
        for _ in 0..frame.params.solve_iterations {
            self.block_solve.record_stream(recorder);
            self.block_apply
                .record_rows(recorder, Count::Dynamic.rows(&frame.params));
        }
    }

    pub(super) fn record_position_iterations(&self, recorder: &mut ComputeRecorder, frame: &Frame) {
        for _ in 0..frame.params.position_iterations {
            self.position_block.record_stream(recorder);
            self.position_apply
                .record_rows(recorder, Count::Dynamic.rows(&frame.params));
        }
    }
}
