use super::stage::{CORE, GEOMETRY, MAX_GRID_WORKGROUPS, Stage, shape_resources, whole};
use crate::dynamics::buffers::{COMPACT_BLOCK, WorldBuffers};
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_CONTACTS, COUNTER_JOINTS, COUNTER_PAIRS};
use dynamis_sort::RadixSort;

pub(super) struct Narrowphase {
    narrowphase: Stage,
    compact_scan: Stage,
    compact_offsets: Stage,
    compact_scatter: Stage,
}

impl Narrowphase {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            narrowphase: Stage::build(
                context,
                "narrowphase",
                include_str!("shaders/narrowphase.wgsl"),
                per_row,
                GEOMETRY,
                &[
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("colliders", whole(&buffers.colliders)),
                    ("pair_major", whole(&buffers.pair_major)),
                    ("pair_minor", whole(&buffers.pair_minor)),
                    ("contacts_raw", whole(&buffers.contacts_raw)),
                    ("contact_valid", whole(&buffers.contact_valid)),
                    ("pair_count", buffers.counter(COUNTER_PAIRS)),
                    ("joint_major", whole(&buffers.joint_filter_major)),
                    ("joint_minor", whole(&buffers.joint_filter_minor)),
                    ("joint_count", buffers.counter(COUNTER_JOINTS)),
                    ("params", whole(&buffers.params)),
                    ("collider_owners", whole(&buffers.collider_owners)),
                ],
                &shape_resources(buffers),
            ),
            compact_scan: Stage::build(
                context,
                "compact_scan",
                include_str!("shaders/compact_scan.wgsl"),
                per_row,
                CORE,
                &[
                    ("valid", whole(&buffers.contact_valid)),
                    ("ranks", whole(&buffers.compact_ranks)),
                    ("block_sums", whole(&buffers.compact_block_sums)),
                    ("count_holder", buffers.counter(COUNTER_PAIRS)),
                ],
                &[],
            ),
            compact_offsets: Stage::build(
                context,
                "compact_offsets",
                include_str!("shaders/compact_offsets.wgsl"),
                per_row,
                CORE,
                &[
                    ("block_sums", whole(&buffers.compact_block_sums)),
                    ("block_offsets", whole(&buffers.compact_block_offsets)),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("pair_count", buffers.counter(COUNTER_PAIRS)),
                ],
                &[],
            ),
            compact_scatter: Stage::build(
                context,
                "compact_scatter",
                include_str!("shaders/compact_scatter.wgsl"),
                per_row,
                CORE,
                &[
                    ("contacts_raw", whole(&buffers.contacts_raw)),
                    ("valid", whole(&buffers.contact_valid)),
                    ("ranks", whole(&buffers.compact_ranks)),
                    ("block_offsets", whole(&buffers.compact_block_offsets)),
                    ("contacts", whole(&buffers.contacts)),
                    ("contact_matched", whole(&buffers.contact_matched)),
                    ("count_holder", buffers.counter(COUNTER_PAIRS)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        sort: &RadixSort,
    ) {
        let words = buffers.collider_words();
        let pairs = buffers.pair_capacity();
        let channels = buffers.sort_lanes_dual(
            buffers.counter(COUNTER_PAIRS),
            &buffers.pair_major,
            &buffers.pair_minor,
        );
        sort.sort(recorder, &channels, words, words, pairs);
        self.narrowphase.record_stride(recorder, pairs);
        self.compact_scan.record_workgroups(
            recorder,
            pairs.div_ceil(COMPACT_BLOCK).min(MAX_GRID_WORKGROUPS),
        );
        self.compact_offsets.record_workgroups(recorder, 1);
        self.compact_scatter.record_stride(recorder, pairs);
    }
}
