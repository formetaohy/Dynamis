use super::FrameParams;
use super::stage::{BLOCKS, CLASS_TOKENS, CORE, OVERFLOW_CORRECTIONS, Program, Stage, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_BLOCKS, COUNTER_CONTACTS, SOLVER_CLASS_COUNT};
use dynamis_sort::RadixSort;
use wgpu::BindGroup;

pub(super) struct Solver {
    reset: Stage,
    total: Stage,
    blocks: Stage,
    boundaries: Stage,
    boundaries_overflow: Stage,
    classify: Stage,
    classify_final: Stage,
    class_scan: Stage,
    class_scatter: Stage,
    overflow_gather: Stage,
    overflow_pairs: Stage,
    class_warm: Stage,
    overflow_warm: Stage,
    overflow_blocks: Stage,
    overflow_apply: Stage,
    class_blocks: Stage,
    position_overflow: Stage,
    position_class: Stage,
    position_overflow_apply: Stage,
    classify_groups: Vec<BindGroup>,
    class_solve_groups: Vec<BindGroup>,
    class_position_groups: Vec<BindGroup>,
}

const CLASSIFY: &str = include_str!("shaders/solver_classify.wgsl");
const CLASS_SOLVE: &str = include_str!("shaders/solver_class.wgsl");
const POSITION_CLASS: &str = include_str!("shaders/position_class.wgsl");

impl Solver {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        let device = context.device();
        let segments = buffers.solver_segments.slot();
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
                ("first_order_bodies", whole(&buffers.solver_a_bodies)),
                ("first_order_blocks", whole(&buffers.solver_a_payload)),
                ("second_order_bodies", whole(&buffers.solver_b_bodies)),
                ("second_order_blocks", whole(&buffers.solver_b_blocks)),
                ("collider_owners", whole(&buffers.collider_owners)),
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
                ("first_a", whole(&buffers.solver_first_a)),
                ("first_b", whole(&buffers.solver_first_b)),
                ("overflow_count", buffers.overflow_count()),
            ],
            &[],
        );
        let boundaries_overflow = Stage::build_entry(
            context,
            Program::entry(
                "solver boundaries overflow",
                include_str!("shaders/solver_boundaries.wgsl"),
                "overflow",
            ),
            per_row,
            CORE,
            &[
                ("segments", segments),
                ("a_bodies", whole(&buffers.solver_a_bodies)),
                ("b_bodies", whole(&buffers.solver_b_bodies)),
                ("first_a", whole(&buffers.solver_overflow_first_a)),
                ("first_b", whole(&buffers.solver_overflow_first_b)),
                ("overflow_count", buffers.overflow_count()),
            ],
            &[],
        );
        let classify_slots = [
            ("class_tokens", whole(&buffers.solver_class_tokens)),
            ("block_first_body", whole(&buffers.solver_block_first_body)),
            (
                "block_second_body",
                whole(&buffers.solver_block_second_body),
            ),
            ("first_order_bodies", whole(&buffers.solver_a_bodies)),
            ("first_order_blocks", whole(&buffers.solver_a_payload)),
            ("second_order_bodies", whole(&buffers.solver_b_bodies)),
            ("second_order_blocks", whole(&buffers.solver_b_blocks)),
            ("first_of_body", whole(&buffers.solver_first_a)),
            ("second_of_body", whole(&buffers.solver_first_b)),
            ("segments", segments),
            ("current_round", buffers.class_round(0)),
            ("class_counts", whole(&buffers.solver_class_counts)),
        ];
        let classify = Stage::build(
            context,
            "solver classify",
            CLASSIFY,
            per_row,
            CLASS_TOKENS,
            &classify_slots,
            &[],
        );
        let classify_final = Stage::build_entry(
            context,
            Program::entry("solver classify settle", CLASSIFY, "settle"),
            per_row,
            CLASS_TOKENS,
            &classify_slots,
            &[],
        );
        let classify_groups = (0..SOLVER_CLASS_COUNT)
            .map(|round| {
                classify.bind_group(
                    device,
                    &[
                        ("class_tokens", whole(&buffers.solver_class_tokens)),
                        ("block_first_body", whole(&buffers.solver_block_first_body)),
                        (
                            "block_second_body",
                            whole(&buffers.solver_block_second_body),
                        ),
                        ("first_order_bodies", whole(&buffers.solver_a_bodies)),
                        ("first_order_blocks", whole(&buffers.solver_a_payload)),
                        ("second_order_bodies", whole(&buffers.solver_b_bodies)),
                        ("second_order_blocks", whole(&buffers.solver_b_blocks)),
                        ("first_of_body", whole(&buffers.solver_first_a)),
                        ("second_of_body", whole(&buffers.solver_first_b)),
                        ("segments", segments),
                        ("current_round", buffers.class_round(round)),
                        ("class_counts", whole(&buffers.solver_class_counts)),
                    ],
                )
            })
            .collect::<Vec<_>>();
        let class_scan = Stage::build(
            context,
            "solver class scan",
            include_str!("shaders/solver_class_scan.wgsl"),
            per_row,
            CORE,
            &[
                ("class_counts", whole(&buffers.solver_class_counts)),
                ("class_bounds", whole(&buffers.solver_class_bounds)),
                ("class_cursors", whole(&buffers.solver_class_cursors)),
                ("overflow_count", buffers.overflow_count()),
            ],
            &[],
        );
        let class_scatter = Stage::build(
            context,
            "solver class scatter",
            include_str!("shaders/solver_class_scatter.wgsl"),
            per_row,
            CLASS_TOKENS,
            &[
                ("class_tokens", whole(&buffers.solver_class_tokens)),
                ("class_cursors", whole(&buffers.solver_class_cursors)),
                ("class_blocks", whole(&buffers.solver_class_blocks)),
                ("segments", segments),
            ],
            &[],
        );
        let overflow_gather = Stage::build(
            context,
            "solver overflow gather",
            include_str!("shaders/solver_overflow_gather.wgsl"),
            per_row,
            CORE,
            &[
                ("class_blocks", whole(&buffers.solver_class_blocks)),
                ("class_bounds", whole(&buffers.solver_class_bounds)),
                ("block_first_body", whole(&buffers.solver_block_first_body)),
                (
                    "block_second_body",
                    whole(&buffers.solver_block_second_body),
                ),
                ("first_order_bodies", whole(&buffers.solver_a_bodies)),
                ("first_order_blocks", whole(&buffers.solver_a_payload)),
                ("segments", segments),
                ("block_counts", whole(&buffers.solver_block_counts)),
                ("contact_counts", whole(&buffers.solver_contact_counts)),
                ("contacts", whole(&buffers.contacts)),
                ("constraint_runtime", whole(&buffers.constraint_runtime)),
                ("overflow_count", buffers.overflow_count()),
            ],
            &[],
        );
        let overflow_pairs = Stage::build(
            context,
            "solver overflow pairs",
            include_str!("shaders/solver_overflow_pairs.wgsl"),
            per_row,
            CORE,
            &[
                ("first_order_blocks", whole(&buffers.solver_a_payload)),
                (
                    "block_second_body",
                    whole(&buffers.solver_block_second_body),
                ),
                ("second_order_bodies", whole(&buffers.solver_b_bodies)),
                ("second_order_blocks", whole(&buffers.solver_b_blocks)),
                ("overflow_count", buffers.overflow_count()),
            ],
            &[],
        );
        let overflow_warm = Stage::build_warm(
            context,
            "solver overflow warm",
            include_str!("shaders/solver_overflow.wgsl"),
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
                ("overflow_count", buffers.overflow_count()),
            ],
            &[],
        );
        let overflow_blocks = Stage::build(
            context,
            "solver overflow blocks",
            include_str!("shaders/solver_overflow.wgsl"),
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
                ("overflow_count", buffers.overflow_count()),
            ],
            &[],
        );
        let overflow_apply = Stage::build(
            context,
            "solver overflow apply",
            include_str!("shaders/solver_overflow_apply.wgsl"),
            per_row,
            CORE,
            &[
                ("params", whole(&buffers.params)),
                ("body_states", whole(&buffers.body_states)),
                ("overflow_first_a", whole(&buffers.solver_overflow_first_a)),
                ("overflow_first_b", whole(&buffers.solver_overflow_first_b)),
                ("a_bodies", whole(&buffers.solver_a_bodies)),
                ("b_bodies", whole(&buffers.solver_b_bodies)),
                ("b_blocks", whole(&buffers.solver_b_blocks)),
                ("block_counts", whole(&buffers.solver_block_counts)),
                ("block_deltas", whole(&buffers.solver_block_deltas)),
                ("overflow_count", buffers.overflow_count()),
            ],
            &[],
        );
        let class_blocks = Stage::build(
            context,
            "solver class blocks",
            CLASS_SOLVE,
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
                ("collider_owners", whole(&buffers.collider_owners)),
                ("class_blocks", whole(&buffers.solver_class_blocks)),
                ("class_range", buffers.class_range(0)),
            ],
            &[],
        );
        let class_warm = Stage::build(
            context,
            "solver class warm",
            include_str!("shaders/solver_class_warm.wgsl"),
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
                ("collider_owners", whole(&buffers.collider_owners)),
                ("class_blocks", whole(&buffers.solver_class_blocks)),
                ("class_range", buffers.class_range(0)),
            ],
            &[],
        );
        let class_solve_groups = (0..SOLVER_CLASS_COUNT)
            .map(|class| {
                class_blocks.bind_group(
                    device,
                    &[
                        ("params", whole(&buffers.params)),
                        ("body_states", whole(&buffers.body_states)),
                        ("body_descs", whole(&buffers.body_descriptors)),
                        ("contacts", whole(&buffers.contacts)),
                        ("constraint_descs", whole(&buffers.constraint_descriptors)),
                        ("constraint_runtime", whole(&buffers.constraint_runtime)),
                        ("wake_flags", whole(&buffers.wake_flags)),
                        ("segments", segments),
                        ("collider_owners", whole(&buffers.collider_owners)),
                        ("class_blocks", whole(&buffers.solver_class_blocks)),
                        ("class_range", buffers.class_range(class)),
                    ],
                )
            })
            .collect::<Vec<_>>();
        let position_overflow = Stage::build(
            context,
            "position overflow",
            include_str!("shaders/position_overflow.wgsl"),
            per_row,
            OVERFLOW_CORRECTIONS,
            &[
                ("params", whole(&buffers.params)),
                ("body_states", whole(&buffers.body_states)),
                ("body_descs", whole(&buffers.body_descriptors)),
                ("contacts", whole(&buffers.contacts)),
                ("segments", segments),
                ("overflow_count", buffers.overflow_count()),
                ("a_payload", whole(&buffers.solver_a_payload)),
                (
                    "block_corrections",
                    whole(&buffers.solver_block_corrections),
                ),
                ("resolution", whole(&buffers.solver_resolution)),
                ("collider_owners", whole(&buffers.collider_owners)),
                ("contributions", whole(&buffers.solver_contributions)),
            ],
            &[],
        );
        let position_class = Stage::build(
            context,
            "position class",
            POSITION_CLASS,
            per_row,
            CORE,
            &[
                ("params", whole(&buffers.params)),
                ("body_states", whole(&buffers.body_states)),
                ("body_descs", whole(&buffers.body_descriptors)),
                ("contacts", whole(&buffers.contacts)),
                ("segments", segments),
                ("resolution", whole(&buffers.solver_resolution)),
                ("collider_owners", whole(&buffers.collider_owners)),
                ("class_blocks", whole(&buffers.solver_class_blocks)),
                ("class_range", buffers.class_range(0)),
            ],
            &[],
        );
        let class_position_groups = (0..SOLVER_CLASS_COUNT)
            .map(|class| {
                position_class.bind_group(
                    device,
                    &[
                        ("params", whole(&buffers.params)),
                        ("body_states", whole(&buffers.body_states)),
                        ("body_descs", whole(&buffers.body_descriptors)),
                        ("contacts", whole(&buffers.contacts)),
                        ("segments", segments),
                        ("resolution", whole(&buffers.solver_resolution)),
                        ("collider_owners", whole(&buffers.collider_owners)),
                        ("class_blocks", whole(&buffers.solver_class_blocks)),
                        ("class_range", buffers.class_range(class)),
                    ],
                )
            })
            .collect::<Vec<_>>();
        let position_overflow_apply = Stage::build(
            context,
            "position overflow apply",
            include_str!("shaders/position_overflow_apply.wgsl"),
            per_row,
            CORE,
            &[
                ("params", whole(&buffers.params)),
                ("body_states", whole(&buffers.body_states)),
                ("overflow_first_a", whole(&buffers.solver_overflow_first_a)),
                ("overflow_first_b", whole(&buffers.solver_overflow_first_b)),
                ("a_bodies", whole(&buffers.solver_a_bodies)),
                ("b_bodies", whole(&buffers.solver_b_bodies)),
                ("b_blocks", whole(&buffers.solver_b_blocks)),
                ("contact_counts", whole(&buffers.solver_contact_counts)),
                (
                    "block_corrections",
                    whole(&buffers.solver_block_corrections),
                ),
                ("overflow_count", buffers.overflow_count()),
                ("resolution", whole(&buffers.solver_resolution)),
                ("contributions", whole(&buffers.solver_contributions)),
            ],
            &[],
        );
        Self {
            reset: Stage::build(
                context,
                "solver_reset",
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
            ),
            total: Stage::build(
                context,
                "solver_total",
                include_str!("shaders/solver_total.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("contacts", whole(&buffers.contacts)),
                    ("constraint_runtime", whole(&buffers.constraint_runtime)),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("segments", segments),
                    ("block_count", buffers.counter(COUNTER_BLOCKS)),
                    ("class_counts", whole(&buffers.solver_class_counts)),
                ],
                &[],
            ),
            blocks,
            boundaries,
            boundaries_overflow,
            classify,
            classify_final,
            class_scan,
            class_scatter,
            overflow_gather,
            overflow_pairs,
            class_warm,
            overflow_warm,
            overflow_blocks,
            overflow_apply,
            class_blocks,
            position_overflow,
            position_class,
            position_overflow_apply,
            classify_groups,
            class_solve_groups,
            class_position_groups,
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
        self.blocks.record_stride(recorder, blocks);
        sort.sort(
            recorder,
            &buffers.sort_lanes(
                buffers.counter(COUNTER_BLOCKS),
                &buffers.solver_a_bodies,
                &buffers.solver_a_payload,
            ),
            words,
            0,
            blocks,
        );
        sort.sort(
            recorder,
            &buffers.sort_lanes(
                buffers.counter(COUNTER_BLOCKS),
                &buffers.solver_b_bodies,
                &buffers.solver_b_blocks,
            ),
            words,
            0,
            blocks,
        );
        self.boundaries.record_stride(recorder, blocks);

        for group in self
            .classify_groups
            .iter()
            .take(SOLVER_CLASS_COUNT as usize - 1)
        {
            self.classify.record_group_stride(recorder, group, blocks);
        }
        self.classify_final.record_group_stride(
            recorder,
            self.classify_groups
                .last()
                .expect("the solver declares at least one class"),
            blocks,
        );
        self.class_scan.record_workgroups(recorder, 1);
        self.class_scatter.record_stride(recorder, blocks);

        self.overflow_gather.record_stride(recorder, blocks);
        sort.sort(
            recorder,
            &buffers.sort_lanes(
                buffers.overflow_count(),
                &buffers.solver_a_bodies,
                &buffers.solver_a_payload,
            ),
            words,
            0,
            blocks,
        );
        self.overflow_pairs.record_stride(recorder, blocks);
        sort.sort(
            recorder,
            &buffers.sort_lanes(
                buffers.overflow_count(),
                &buffers.solver_b_bodies,
                &buffers.solver_b_blocks,
            ),
            words,
            0,
            blocks,
        );
        self.boundaries_overflow.record_stride(recorder, blocks);
        for group in &self.class_solve_groups {
            self.class_warm.record_group_stride(recorder, group, blocks);
        }
        self.overflow_warm.record_warm_stride(recorder, blocks);
        self.overflow_apply.record(recorder, params.dynamic_count);
        for _ in 0..params.solve_iterations {
            for group in &self.class_solve_groups {
                self.class_blocks
                    .record_group_stride(recorder, group, blocks);
            }
            self.overflow_blocks.record_stride(recorder, blocks);
            self.overflow_apply.record(recorder, params.dynamic_count);
        }
    }

    fn record_position(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        params: &FrameParams,
    ) {
        for group in &self.class_position_groups {
            self.position_class
                .record_group_stride(recorder, group, buffers.block_capacity());
        }
        self.position_overflow
            .record_stride(recorder, buffers.block_capacity());
        self.position_overflow_apply
            .record(recorder, params.dynamic_count);
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
