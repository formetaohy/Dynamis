use super::buffers::{COMPACT_BLOCK, RigidBuffers};
use super::shader;
use super::shader::{CORE, GEOMETRY};
use crate::dynamics::engine::{MAX_DISPATCH_WORKGROUPS, Stage, whole};
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
    pub(super) fn build(context: &GpuContext, buffers: &RigidBuffers) -> Self {
        Self {
            narrowphase: Stage::build(
                context,
                "narrowphase",
                shader::stream(
                    context,
                    include_str!("shaders/narrowphase.wgsl"),
                    GEOMETRY,
                    "work",
                    buffers.pair_capacity(),
                ),
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
                &buffers.shape_resources(),
            ),
            compact_scan: Stage::build(
                context,
                "compact_scan",
                shader::workgroups(context, include_str!("shaders/compact_scan.wgsl"), CORE),
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
                shader::workgroups(context, include_str!("shaders/compact_offsets.wgsl"), CORE),
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
                shader::stream(
                    context,
                    include_str!("shaders/compact_scatter.wgsl"),
                    CORE,
                    "work",
                    buffers.pair_capacity(),
                ),
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
        buffers: &RigidBuffers,
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
        self.narrowphase.record_stream(recorder);
        self.compact_scan.record_workgroups(
            recorder,
            pairs.div_ceil(COMPACT_BLOCK).min(MAX_DISPATCH_WORKGROUPS),
        );
        self.compact_offsets.record_workgroups(recorder, 1);
        self.compact_scatter.record_stream(recorder);
    }
}
