use super::Count;
use super::streams::RigidStream;
use crate::dynamics::Frame;
use crate::dynamics::engine::Stage;
use crate::dynamics::scene::SceneStream;
use crate::dynamics::shader;
use crate::dynamics::shader::GRID_INDEX;
use crate::dynamics::streams::Streams;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_ENTRIES;
use dynamis_sort::RadixSort;

pub(super) struct Broadphase {
    cell_pairs: Stage,
    level_links: Stage,
}

impl Broadphase {
    pub(super) fn build(context: &GpuContext, streams: &Streams) -> Self {
        Self {
            cell_pairs: Stage::build(
                context,
                "cell_pairs",
                shader::stream(
                    context,
                    include_str!("../shaders/cell_pairs.wgsl"),
                    GRID_INDEX,
                    "work",
                    RigidStream::GridEntryKeys,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("pair_major", RigidStream::PairMajor.whole()),
                    ("pair_minor", RigidStream::PairMinor.whole()),
                    ("body_activity", RigidStream::BodyActivity.whole()),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
                    ("aabbs", RigidStream::ColliderAabbs.whole()),
                    ("entry_keys", RigidStream::GridEntryKeys.whole()),
                    ("entry_colliders", RigidStream::GridEntryColliders.whole()),
                    ("counters", SceneStream::Counters.whole()),
                ],
                &[],
            ),
            level_links: Stage::build(
                context,
                "level_links",
                shader::rows(
                    context,
                    include_str!("../shaders/level_links.wgsl"),
                    GRID_INDEX,
                    Count::Colliders,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("pair_major", RigidStream::PairMajor.whole()),
                    ("pair_minor", RigidStream::PairMinor.whole()),
                    ("aabbs", RigidStream::ColliderAabbs.whole()),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
                    ("body_activity", RigidStream::BodyActivity.whole()),
                    ("entry_keys", RigidStream::GridEntryKeys.whole()),
                    ("entry_colliders", RigidStream::GridEntryColliders.whole()),
                    ("counters", SceneStream::Counters.whole()),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &Streams,
        frame: &Frame,
        sort: &RadixSort,
    ) {
        let channels = streams.sort_lanes(
            streams.scene.counter(COUNTER_ENTRIES),
            RigidStream::GridEntryKeys.whole(),
            RigidStream::GridEntryColliders.whole(),
        );
        sort.sort(recorder, &channels, 4, 0, streams.rigid.entry_capacity());
        self.cell_pairs.record_stream(recorder, streams);
        self.level_links
            .record_rows(recorder, streams, Count::Colliders.rows(&frame.params));
    }
}
