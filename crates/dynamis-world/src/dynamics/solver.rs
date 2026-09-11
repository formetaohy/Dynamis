use super::FrameParams;
use super::dispatch::{
    SOLVER_BLOCKS, SOLVER_BOUNDARIES, SOLVER_COUNT, SOLVER_GATHER_B, SOLVER_POSITION,
    SORT_SOLVER_A, SORT_SOLVER_B,
};
use super::stage::{BLOCKS, CORE, CORRECTIONS, RO, RW, Stage, UNIFORM, whole};
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
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.solver.first_a)),
                    (RW, whole(&buffers.solver.first_b)),
                    (RW, whole(&buffers.solver.counts)),
                    (RW, whole(&buffers.solver.contact_counts)),
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
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.contacts.manifolds)),
                    (RO, whole(&buffers.constraints.runtime)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RW, whole(&buffers.solver.segments)),
                    (RW, buffers.counter(COUNTER_BLOCKS)),
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
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RW, whole(&buffers.contacts.manifolds)),
                    (RO, whole(&buffers.constraints.descriptors)),
                    (RO, whole(&buffers.constraints.runtime)),
                    (RO, whole(&buffers.solver.segments)),
                    (RW, whole(&buffers.solver.a_bodies)),
                    (RW, whole(&buffers.solver.a_payload)),
                    (RW, whole(&buffers.solver.counts)),
                    (RW, whole(&buffers.solver.contact_counts)),
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
                    (RO, whole(&buffers.contacts.manifolds)),
                    (RO, whole(&buffers.constraints.descriptors)),
                    (RO, whole(&buffers.solver.segments)),
                    (RO, whole(&buffers.solver.a_payload)),
                    (RW, whole(&buffers.solver.b_bodies)),
                    (RW, whole(&buffers.solver.b_blocks)),
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
                    (RO, whole(&buffers.solver.segments)),
                    (RO, whole(&buffers.solver.a_bodies)),
                    (RO, whole(&buffers.solver.b_bodies)),
                    (RW, whole(&buffers.solver.first_a)),
                    (RW, whole(&buffers.solver.first_b)),
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
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RW, whole(&buffers.contacts.manifolds)),
                    (RO, whole(&buffers.constraints.descriptors)),
                    (RW, whole(&buffers.constraints.runtime)),
                    (RW, whole(&buffers.islands.wake_flags)),
                    (RO, whole(&buffers.solver.segments)),
                    (RO, whole(&buffers.solver.a_payload)),
                    (RW, whole(&buffers.solver.deltas)),
                    (RO, whole(&buffers.solver.counts)),
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
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.solver.first_a)),
                    (RO, whole(&buffers.solver.first_b)),
                    (RO, whole(&buffers.solver.a_bodies)),
                    (RO, whole(&buffers.solver.b_bodies)),
                    (RO, whole(&buffers.solver.b_blocks)),
                    (RO, whole(&buffers.solver.counts)),
                    (RO, whole(&buffers.solver.deltas)),
                    (RW, buffers.counter(COUNTER_BLOCKS)),
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
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RO, whole(&buffers.contacts.manifolds)),
                    (RO, whole(&buffers.solver.segments)),
                    (RO, whole(&buffers.solver.a_payload)),
                    (RW, whole(&buffers.solver.corrections)),
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
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.solver.first_a)),
                    (RO, whole(&buffers.solver.first_b)),
                    (RO, whole(&buffers.solver.a_bodies)),
                    (RO, whole(&buffers.solver.b_bodies)),
                    (RO, whole(&buffers.solver.b_blocks)),
                    (RO, whole(&buffers.solver.contact_counts)),
                    (RO, whole(&buffers.solver.corrections)),
                    (RW, buffers.counter(COUNTER_BLOCKS)),
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
        self.count
            .record_indirect(recorder, &buffers.dispatch, SOLVER_COUNT);
        let words = buffers.body_words();
        let channels = buffers.sort_lanes(
            buffers.counter(COUNTER_BLOCKS),
            &buffers.solver.a_bodies,
            &buffers.solver.a_payload,
        );
        sort.sort(
            recorder,
            &channels,
            words,
            0,
            &buffers.dispatch,
            SORT_SOLVER_A,
        );

        self.gather_b
            .record_indirect(recorder, &buffers.dispatch, SOLVER_GATHER_B);
        let channels = buffers.sort_lanes(
            buffers.counter(COUNTER_BLOCKS),
            &buffers.solver.b_bodies,
            &buffers.solver.b_blocks,
        );
        sort.sort(
            recorder,
            &channels,
            words,
            0,
            &buffers.dispatch,
            SORT_SOLVER_B,
        );

        self.boundaries
            .record_indirect(recorder, &buffers.dispatch, SOLVER_BOUNDARIES);
        self.blocks
            .record_warm_indirect(recorder, &buffers.dispatch, SOLVER_BLOCKS);
        self.apply.record(recorder, params.dynamic_count);

        for _ in 0..params.solve_iterations {
            self.blocks
                .record_indirect(recorder, &buffers.dispatch, SOLVER_BLOCKS);
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
            .record_indirect(recorder, &buffers.dispatch, SOLVER_POSITION);
        self.position_apply.record(recorder, params.dynamic_count);
    }
}
