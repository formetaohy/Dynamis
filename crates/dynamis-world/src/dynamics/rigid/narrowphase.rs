use super::buffers::{COMPACT_BLOCK, RigidBuffers, StreamId};
use super::shader;
use super::shader::{CORE, GEOMETRY};
use crate::dynamics::engine::{MAX_DISPATCH_WORKGROUPS, Stage};
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
                    StreamId::PairMajor,
                ),
                buffers,
                &[
                    ("body_states", StreamId::BodyStates.whole()),
                    ("body_descs", StreamId::BodyDescriptors.whole()),
                    ("colliders", StreamId::Colliders.whole()),
                    ("pair_major", StreamId::PairMajor.whole()),
                    ("pair_minor", StreamId::PairMinor.whole()),
                    ("contacts_raw", StreamId::ContactsRaw.whole()),
                    ("contact_valid", StreamId::ContactValid.whole()),
                    ("pair_count", buffers.counter(COUNTER_PAIRS)),
                    ("joint_major", StreamId::JointFilterMajor.whole()),
                    ("joint_minor", StreamId::JointFilterMinor.whole()),
                    ("joint_count", buffers.counter(COUNTER_JOINTS)),
                    ("params", StreamId::Params.whole()),
                    ("collider_owners", StreamId::ColliderOwners.whole()),
                ],
                &RigidBuffers::shape_resources(),
            ),
            compact_scan: Stage::build(
                context,
                "compact_scan",
                shader::workgroups(context, include_str!("shaders/compact_scan.wgsl"), CORE),
                buffers,
                &[
                    ("valid", StreamId::ContactValid.whole()),
                    ("ranks", StreamId::CompactRanks.whole()),
                    ("block_sums", StreamId::CompactBlockSums.whole()),
                    ("count_holder", buffers.counter(COUNTER_PAIRS)),
                ],
                &[],
            ),
            compact_offsets: Stage::build(
                context,
                "compact_offsets",
                shader::workgroups(context, include_str!("shaders/compact_offsets.wgsl"), CORE),
                buffers,
                &[
                    ("block_sums", StreamId::CompactBlockSums.whole()),
                    ("block_offsets", StreamId::CompactBlockOffsets.whole()),
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
                    StreamId::PairMajor,
                ),
                buffers,
                &[
                    ("contacts_raw", StreamId::ContactsRaw.whole()),
                    ("valid", StreamId::ContactValid.whole()),
                    ("ranks", StreamId::CompactRanks.whole()),
                    ("block_offsets", StreamId::CompactBlockOffsets.whole()),
                    ("contacts", StreamId::Contacts.whole()),
                    ("contact_matched", StreamId::ContactMatched.whole()),
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
            StreamId::PairMajor.whole(),
            StreamId::PairMinor.whole(),
        );
        sort.sort(recorder, &channels, words, words, pairs);
        self.narrowphase.record_stream(recorder, buffers);
        self.compact_scan.record_workgroups(
            recorder,
            buffers,
            pairs.div_ceil(COMPACT_BLOCK).min(MAX_DISPATCH_WORKGROUPS),
        );
        self.compact_offsets.record_workgroups(recorder, buffers, 1);
        self.compact_scatter.record_stream(recorder, buffers);
    }
}
