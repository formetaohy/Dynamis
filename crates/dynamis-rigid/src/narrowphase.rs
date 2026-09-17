use super::streams::COMPACT_BLOCK;
use super::streams::RigidStream;
use crate::RigidFrame;
use crate::sort;
use dynamis_abi::{COUNTER_CONTACTS, COUNTER_JOINTS, COUNTER_PAIRS, COUNTER_REFUSED_CONTACTS};
use dynamis_broadphase::{BroadphaseStream, pair_capacity};
use dynamis_gpu::ResourceSource;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{MAX_DISPATCH_WORKGROUPS, PassRuntime, Stage};
use dynamis_shader::{CORE, Extent, stream, workgroups};
use dynamis_sort::RadixSort;
use dynamis_state::StateStream;

pub struct Narrowphase {
    narrowphase: Stage,
    compact_scan: Stage,
    compact_offsets: Stage,
    compact_scatter: Stage,
    sort: RadixSort,
}

impl PassRuntime<RigidFrame> for Narrowphase {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            narrowphase: Stage::build(
                context,
                "narrowphase",
                stream(
                    context,
                    include_str!("../shaders/narrowphase.wgsl"),
                    &crate::geometry_fragments(),
                    "work",
                    Extent::slot(COUNTER_PAIRS, "pair_count", "pair_major"),
                ),
                streams,
                &[
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("colliders", StateStream::Colliders.whole()),
                    ("pair_major", BroadphaseStream::PairMajor.whole()),
                    ("pair_minor", BroadphaseStream::PairMinor.whole()),
                    ("contacts_raw", RigidStream::ContactsRaw.whole()),
                    ("contact_valid", RigidStream::ContactValid.whole()),
                    ("pair_count", dynamis_state::counter(COUNTER_PAIRS)),
                    ("joint_major", RigidStream::JointFilterMajor.whole()),
                    ("joint_minor", RigidStream::JointFilterMinor.whole()),
                    ("joint_count", dynamis_state::counter(COUNTER_JOINTS)),
                    ("params", StateStream::Params.whole()),
                    ("collider_owners", StateStream::ColliderOwners.whole()),
                ],
                &dynamis_state::shape_resources(),
            ),
            compact_scan: Stage::build(
                context,
                "compact_scan",
                workgroups(context, include_str!("../shaders/compact_scan.wgsl"), CORE),
                streams,
                &[
                    ("valid", RigidStream::ContactValid.whole()),
                    ("ranks", RigidStream::CompactRanks.whole()),
                    ("block_sums", RigidStream::CompactBlockSums.whole()),
                    ("count_holder", dynamis_state::counter(COUNTER_PAIRS)),
                ],
                &[],
            ),
            compact_offsets: Stage::build(
                context,
                "compact_offsets",
                workgroups(
                    context,
                    include_str!("../shaders/compact_offsets.wgsl"),
                    CORE,
                ),
                streams,
                &[
                    ("block_sums", RigidStream::CompactBlockSums.whole()),
                    ("block_offsets", RigidStream::CompactBlockOffsets.whole()),
                    ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
                    ("pair_count", dynamis_state::counter(COUNTER_PAIRS)),
                ],
                &[],
            ),
            compact_scatter: Stage::build(
                context,
                "compact_scatter",
                stream(
                    context,
                    include_str!("../shaders/compact_scatter.wgsl"),
                    CORE,
                    "work",
                    Extent::slot(COUNTER_PAIRS, "count_holder", "contacts_raw"),
                ),
                streams,
                &[
                    ("contacts_raw", RigidStream::ContactsRaw.whole()),
                    ("valid", RigidStream::ContactValid.whole()),
                    ("ranks", RigidStream::CompactRanks.whole()),
                    ("block_offsets", RigidStream::CompactBlockOffsets.whole()),
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_matched", RigidStream::ContactMatched.whole()),
                    ("count_holder", dynamis_state::counter(COUNTER_PAIRS)),
                    ("refused", dynamis_state::counter(COUNTER_REFUSED_CONTACTS)),
                ],
                &[],
            ),
            sort: RadixSort::new(context),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        let words = frame.shape.collider_slot_words;
        let pairs = pair_capacity(streams);
        let channels = sort::lanes_dual(
            streams,
            dynamis_state::counter(COUNTER_PAIRS),
            BroadphaseStream::PairMajor.whole(),
            BroadphaseStream::PairMinor.whole(),
        );
        self.sort.sort(recorder, &channels, words, words);
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
