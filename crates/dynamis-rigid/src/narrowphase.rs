use super::streams::COMPACT_BLOCK;
use super::streams::RigidStream;
use crate::sort;
use dynamis_abi::{COUNTER_CONTACTS, COUNTER_JOINTS, COUNTER_PAIRS};
use dynamis_broadphase::{BroadphaseStream, pair_capacity};
use dynamis_engine::Resources;
use dynamis_engine::{MAX_DISPATCH_WORKGROUPS, Stage};
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernels::{CORE, GEOMETRY, stream, workgroups};
use dynamis_scene::SceneStream;
use dynamis_sort::RadixSort;

pub struct Narrowphase {
    narrowphase: Stage,
    compact_scan: Stage,
    compact_offsets: Stage,
    compact_scatter: Stage,
}

impl Narrowphase {
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            narrowphase: Stage::build(
                context,
                "narrowphase",
                stream(
                    context,
                    include_str!("../shaders/narrowphase.wgsl"),
                    GEOMETRY,
                    "work",
                    BroadphaseStream::PairMajor,
                ),
                streams,
                &[
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("colliders", SceneStream::Colliders.whole()),
                    ("pair_major", BroadphaseStream::PairMajor.whole()),
                    ("pair_minor", BroadphaseStream::PairMinor.whole()),
                    ("contacts_raw", RigidStream::ContactsRaw.whole()),
                    ("contact_valid", RigidStream::ContactValid.whole()),
                    ("pair_count", dynamis_scene::counter(COUNTER_PAIRS)),
                    ("joint_major", RigidStream::JointFilterMajor.whole()),
                    ("joint_minor", RigidStream::JointFilterMinor.whole()),
                    ("joint_count", dynamis_scene::counter(COUNTER_JOINTS)),
                    ("params", SceneStream::Params.whole()),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
                ],
                &dynamis_scene::shape_resources(),
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
                    ("count_holder", dynamis_scene::counter(COUNTER_PAIRS)),
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
                    ("contact_count", dynamis_scene::counter(COUNTER_CONTACTS)),
                    ("pair_count", dynamis_scene::counter(COUNTER_PAIRS)),
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
                    BroadphaseStream::PairMajor,
                ),
                streams,
                &[
                    ("contacts_raw", RigidStream::ContactsRaw.whole()),
                    ("valid", RigidStream::ContactValid.whole()),
                    ("ranks", RigidStream::CompactRanks.whole()),
                    ("block_offsets", RigidStream::CompactBlockOffsets.whole()),
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_matched", RigidStream::ContactMatched.whole()),
                    ("count_holder", dynamis_scene::counter(COUNTER_PAIRS)),
                ],
                &[],
            ),
        }
    }

    pub fn record(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        sort: &RadixSort,
    ) {
        let words = dynamis_scene::collider_words(streams);
        let pairs = pair_capacity(streams);
        let channels = sort::lanes_dual(
            streams,
            dynamis_scene::counter(COUNTER_PAIRS),
            BroadphaseStream::PairMajor.whole(),
            BroadphaseStream::PairMinor.whole(),
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
