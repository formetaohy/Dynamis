use super::shader;
use super::shader::{CORE, GEOMETRY};
use super::streams::COMPACT_BLOCK;
use super::streams::RigidStream;
use crate::dynamics::engine::{MAX_DISPATCH_WORKGROUPS, Stage};
use crate::dynamics::scene::SceneStream;
use crate::dynamics::streams::Streams;
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
    pub(super) fn build(context: &GpuContext, streams: &Streams) -> Self {
        Self {
            narrowphase: Stage::build(
                context,
                "narrowphase",
                shader::stream(
                    context,
                    include_str!("shaders/narrowphase.wgsl"),
                    GEOMETRY,
                    "work",
                    RigidStream::PairMajor,
                ),
                streams,
                &[
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("colliders", SceneStream::Colliders.whole()),
                    ("pair_major", RigidStream::PairMajor.whole()),
                    ("pair_minor", RigidStream::PairMinor.whole()),
                    ("contacts_raw", RigidStream::ContactsRaw.whole()),
                    ("contact_valid", RigidStream::ContactValid.whole()),
                    ("pair_count", streams.scene.counter(COUNTER_PAIRS)),
                    ("joint_major", RigidStream::JointFilterMajor.whole()),
                    ("joint_minor", RigidStream::JointFilterMinor.whole()),
                    ("joint_count", streams.scene.counter(COUNTER_JOINTS)),
                    ("params", SceneStream::Params.whole()),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
                ],
                &streams.scene.shape_resources(),
            ),
            compact_scan: Stage::build(
                context,
                "compact_scan",
                shader::workgroups(context, include_str!("shaders/compact_scan.wgsl"), CORE),
                streams,
                &[
                    ("valid", RigidStream::ContactValid.whole()),
                    ("ranks", RigidStream::CompactRanks.whole()),
                    ("block_sums", RigidStream::CompactBlockSums.whole()),
                    ("count_holder", streams.scene.counter(COUNTER_PAIRS)),
                ],
                &[],
            ),
            compact_offsets: Stage::build(
                context,
                "compact_offsets",
                shader::workgroups(context, include_str!("shaders/compact_offsets.wgsl"), CORE),
                streams,
                &[
                    ("block_sums", RigidStream::CompactBlockSums.whole()),
                    ("block_offsets", RigidStream::CompactBlockOffsets.whole()),
                    ("contact_count", streams.scene.counter(COUNTER_CONTACTS)),
                    ("pair_count", streams.scene.counter(COUNTER_PAIRS)),
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
                    RigidStream::PairMajor,
                ),
                streams,
                &[
                    ("contacts_raw", RigidStream::ContactsRaw.whole()),
                    ("valid", RigidStream::ContactValid.whole()),
                    ("ranks", RigidStream::CompactRanks.whole()),
                    ("block_offsets", RigidStream::CompactBlockOffsets.whole()),
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_matched", RigidStream::ContactMatched.whole()),
                    ("count_holder", streams.scene.counter(COUNTER_PAIRS)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &Streams,
        sort: &RadixSort,
    ) {
        let words = streams.scene.collider_words();
        let pairs = streams.rigid.pair_capacity();
        let channels = streams.sort_lanes_dual(
            streams.scene.counter(COUNTER_PAIRS),
            RigidStream::PairMajor.whole(),
            RigidStream::PairMinor.whole(),
        );
        sort.sort(recorder, &channels, words, words, pairs);
        self.narrowphase.record_stream(recorder, streams);
        self.compact_scan.record_workgroups(
            recorder,
            streams,
            pairs.div_ceil(COMPACT_BLOCK).min(MAX_DISPATCH_WORKGROUPS),
        );
        self.compact_offsets.record_workgroups(recorder, streams, 1);
        self.compact_scatter.record_stream(recorder, streams);
    }
}
