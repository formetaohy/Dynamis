use super::Count;
use super::Frame;
use super::buffers::{RigidBuffers, StreamId};
use super::shader;
use super::shader::GRID_INDEX;
use crate::dynamics::engine::Stage;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_ENTRIES;
use dynamis_sort::RadixSort;

pub(super) struct Broadphase {
    cell_pairs: Stage,
    level_links: Stage,
}

impl Broadphase {
    pub(super) fn build(context: &GpuContext, buffers: &RigidBuffers) -> Self {
        Self {
            cell_pairs: Stage::build(
                context,
                "cell_pairs",
                shader::stream(
                    context,
                    include_str!("shaders/cell_pairs.wgsl"),
                    GRID_INDEX,
                    "work",
                    StreamId::GridEntryKeys,
                ),
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("pair_major", StreamId::PairMajor.whole()),
                    ("pair_minor", StreamId::PairMinor.whole()),
                    ("body_activity", StreamId::BodyActivity.whole()),
                    ("collider_owners", StreamId::ColliderOwners.whole()),
                    ("aabbs", StreamId::ColliderAabbs.whole()),
                    ("entry_keys", StreamId::GridEntryKeys.whole()),
                    ("entry_colliders", StreamId::GridEntryColliders.whole()),
                    ("counters", StreamId::Counters.whole()),
                ],
                &[],
            ),
            level_links: Stage::build(
                context,
                "level_links",
                shader::rows(
                    context,
                    include_str!("shaders/level_links.wgsl"),
                    GRID_INDEX,
                    Count::Colliders,
                ),
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("pair_major", StreamId::PairMajor.whole()),
                    ("pair_minor", StreamId::PairMinor.whole()),
                    ("aabbs", StreamId::ColliderAabbs.whole()),
                    ("collider_owners", StreamId::ColliderOwners.whole()),
                    ("body_activity", StreamId::BodyActivity.whole()),
                    ("entry_keys", StreamId::GridEntryKeys.whole()),
                    ("entry_colliders", StreamId::GridEntryColliders.whole()),
                    ("counters", StreamId::Counters.whole()),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        frame: &Frame,
        sort: &RadixSort,
    ) {
        let channels = buffers.sort_lanes(
            buffers.counter(COUNTER_ENTRIES),
            StreamId::GridEntryKeys.whole(),
            StreamId::GridEntryColliders.whole(),
        );
        sort.sort(recorder, &channels, 4, 0, buffers.entry_capacity());
        self.cell_pairs.record_stream(recorder, buffers);
        self.level_links
            .record_rows(recorder, buffers, Count::Colliders.rows(&frame.params));
    }
}
