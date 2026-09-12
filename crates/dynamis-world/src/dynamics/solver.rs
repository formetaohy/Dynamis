use super::FrameParams;
use super::stage::{BLOCKS, CORE, CORRECTIONS, Stage, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_BLOCKS, COUNTER_CONTACTS};
use dynamis_sort::RadixSort;

pub(super) struct Solver {
    reset: Stage,
    total: Stage,
    count: Stage,
    gather_b: Stage,
    boundaries: Stage,
    blocks: Stage,
    apply: Stage,
    position: Stage,
    position_apply: Stage,
}

impl Solver {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            reset: Stage::build(
                context,
                "solver_reset",
                include_str!("shaders/solver_reset.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("first_a", whole(&buffers.solver.first_a)),
                    ("first_b", whole(&buffers.solver.first_b)),
                    ("block_counts", whole(&buffers.solver.counts)),
                    ("contact_counts", whole(&buffers.solver.contact_counts)),
                    ("resolution", whole(&buffers.solver.resolution)),
                ],
                &[],
            ),
            total: Stage::build(
                context,
                "solver_total",
                include_str!("shaders/solver_total.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("contacts", whole(&buffers.contacts.manifolds)),
                    ("constraint_runtime", whole(&buffers.constraints.runtime)),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("segments", whole(&buffers.solver.segments)),
                    ("block_count", buffers.counter(COUNTER_BLOCKS)),
                ],
                &[],
            ),
            count: Stage::build(
                context,
                "solver_count",
                include_str!("shaders/solver_count.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.bodies.states)),
                    ("body_descs", whole(&buffers.bodies.descriptors)),
                    ("contacts", whole(&buffers.contacts.manifolds)),
                    ("constraint_descs", whole(&buffers.constraints.descriptors)),
                    ("constraint_runtime", whole(&buffers.constraints.runtime)),
                    ("segments", whole(&buffers.solver.segments)),
                    ("a_bodies", whole(&buffers.solver.a_bodies)),
                    ("a_payload", whole(&buffers.solver.a_payload)),
                    ("block_counts", whole(&buffers.solver.counts)),
                    ("contact_counts", whole(&buffers.solver.contact_counts)),
                ],
                &[],
            ),
            gather_b: Stage::build(
                context,
                "solver_gather_b",
                include_str!("shaders/solver_gather_b.wgsl"),
                per_row,
                CORE,
                &[
                    ("contacts", whole(&buffers.contacts.manifolds)),
                    ("constraint_descs", whole(&buffers.constraints.descriptors)),
                    ("segments", whole(&buffers.solver.segments)),
                    ("a_payload", whole(&buffers.solver.a_payload)),
                    ("b_bodies", whole(&buffers.solver.b_bodies)),
                    ("b_blocks", whole(&buffers.solver.b_blocks)),
                ],
                &[],
            ),
            boundaries: Stage::build(
                context,
                "solver_boundaries",
                include_str!("shaders/solver_boundaries.wgsl"),
                per_row,
                CORE,
                &[
                    ("segments", whole(&buffers.solver.segments)),
                    ("a_bodies", whole(&buffers.solver.a_bodies)),
                    ("b_bodies", whole(&buffers.solver.b_bodies)),
                    ("first_a", whole(&buffers.solver.first_a)),
                    ("first_b", whole(&buffers.solver.first_b)),
                ],
                &[],
            ),
            blocks: Stage::build_warm(
                context,
                "solver_blocks",
                include_str!("shaders/solver.wgsl"),
                per_row,
                BLOCKS,
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.bodies.states)),
                    ("body_descs", whole(&buffers.bodies.descriptors)),
                    ("contacts", whole(&buffers.contacts.manifolds)),
                    ("constraint_descs", whole(&buffers.constraints.descriptors)),
                    ("constraint_runtime", whole(&buffers.constraints.runtime)),
                    ("wake_flags", whole(&buffers.islands.wake_flags)),
                    ("segments", whole(&buffers.solver.segments)),
                    ("a_payload", whole(&buffers.solver.a_payload)),
                    ("block_deltas", whole(&buffers.solver.deltas)),
                    ("block_counts", whole(&buffers.solver.counts)),
                ],
                &[],
            ),
            apply: Stage::build(
                context,
                "solver_apply",
                include_str!("shaders/solver_apply.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.bodies.states)),
                    ("first_a", whole(&buffers.solver.first_a)),
                    ("first_b", whole(&buffers.solver.first_b)),
                    ("a_bodies", whole(&buffers.solver.a_bodies)),
                    ("b_bodies", whole(&buffers.solver.b_bodies)),
                    ("b_blocks", whole(&buffers.solver.b_blocks)),
                    ("block_counts", whole(&buffers.solver.counts)),
                    ("block_deltas", whole(&buffers.solver.deltas)),
                    ("block_count", buffers.counter(COUNTER_BLOCKS)),
                ],
                &[],
            ),
            position: Stage::build(
                context,
                "position",
                include_str!("shaders/position.wgsl"),
                per_row,
                CORRECTIONS,
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.bodies.states)),
                    ("body_descs", whole(&buffers.bodies.descriptors)),
                    ("contacts", whole(&buffers.contacts.manifolds)),
                    ("segments", whole(&buffers.solver.segments)),
                    ("a_payload", whole(&buffers.solver.a_payload)),
                    ("block_corrections", whole(&buffers.solver.corrections)),
                    ("resolution", whole(&buffers.solver.resolution)),
                ],
                &[],
            ),
            position_apply: Stage::build(
                context,
                "position_apply",
                include_str!("shaders/position_apply.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.bodies.states)),
                    ("first_a", whole(&buffers.solver.first_a)),
                    ("first_b", whole(&buffers.solver.first_b)),
                    ("a_bodies", whole(&buffers.solver.a_bodies)),
                    ("b_bodies", whole(&buffers.solver.b_bodies)),
                    ("b_blocks", whole(&buffers.solver.b_blocks)),
                    ("contact_counts", whole(&buffers.solver.contact_counts)),
                    ("block_corrections", whole(&buffers.solver.corrections)),
                    ("resolution", whole(&buffers.solver.resolution)),
                    ("block_count", buffers.counter(COUNTER_BLOCKS)),
                ],
                &[],
            ),
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
        self.count.record_stride(recorder, blocks);
        let words = buffers.body_words();
        let channels = buffers.sort_lanes(
            buffers.counter(COUNTER_BLOCKS),
            &buffers.solver.a_bodies,
            &buffers.solver.a_payload,
        );
        sort.sort(recorder, &channels, words, 0, blocks);

        self.gather_b.record_stride(recorder, blocks);
        let channels = buffers.sort_lanes(
            buffers.counter(COUNTER_BLOCKS),
            &buffers.solver.b_bodies,
            &buffers.solver.b_blocks,
        );
        sort.sort(recorder, &channels, words, 0, blocks);

        self.boundaries.record_stride(recorder, blocks);
        self.blocks.record_warm_stride(recorder, blocks);
        self.apply.record(recorder, params.dynamic_count);

        for _ in 0..params.solve_iterations {
            self.blocks.record_stride(recorder, blocks);
            self.apply.record(recorder, params.dynamic_count);
        }
    }

    pub(super) fn record_position(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        params: &FrameParams,
    ) {
        self.position
            .record_stride(recorder, buffers.block_capacity());
        self.position_apply.record(recorder, params.dynamic_count);
    }

    pub(super) fn record_position_iterations(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        params: &FrameParams,
    ) {
        for _ in 0..params.position_iterations {
            self.record_position(recorder, buffers, params);
        }
    }
}
