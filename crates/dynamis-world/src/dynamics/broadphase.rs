use super::Frame;
use super::stage::{Count, Coverage, GRID_INDEX, Slots, Stage, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_ENTRIES;
use dynamis_sort::RadixSort;

pub(super) struct Broadphase {
    cell_pairs: Stage,
    level_links: Stage,
}

impl Broadphase {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers) -> Self {
        Self {
            cell_pairs: Stage::build(
                context,
                "cell_pairs",
                include_str!("shaders/cell_pairs.wgsl"),
                GRID_INDEX,
                Coverage::Stream {
                    kernel: "work",
                    slots: Slots::Entries,
                },
                &[
                    ("params", whole(&buffers.params)),
                    ("pair_major", whole(&buffers.pair_major)),
                    ("pair_minor", whole(&buffers.pair_minor)),
                    ("body_activity", whole(&buffers.body_activity)),
                    ("collider_owners", whole(&buffers.collider_owners)),
                    ("aabbs", whole(&buffers.collider_aabbs)),
                    ("entry_keys", whole(&buffers.grid_entry_keys)),
                    ("entry_colliders", whole(&buffers.grid_entry_colliders)),
                    ("counters", whole(&buffers.counters)),
                ],
                &[],
            ),
            level_links: Stage::build(
                context,
                "level_links",
                include_str!("shaders/level_links.wgsl"),
                GRID_INDEX,
                Coverage::Live(Count::Colliders),
                &[
                    ("params", whole(&buffers.params)),
                    ("pair_major", whole(&buffers.pair_major)),
                    ("pair_minor", whole(&buffers.pair_minor)),
                    ("aabbs", whole(&buffers.collider_aabbs)),
                    ("collider_owners", whole(&buffers.collider_owners)),
                    ("body_activity", whole(&buffers.body_activity)),
                    ("entry_keys", whole(&buffers.grid_entry_keys)),
                    ("entry_colliders", whole(&buffers.grid_entry_colliders)),
                    ("counters", whole(&buffers.counters)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        frame: &Frame,
        sort: &RadixSort,
    ) {
        let channels = buffers.sort_lanes(
            buffers.counter(COUNTER_ENTRIES),
            &buffers.grid_entry_keys,
            &buffers.grid_entry_colliders,
        );
        sort.sort(recorder, &channels, 4, 0, buffers.entry_capacity());
        self.cell_pairs.record(recorder, buffers, frame);
        self.level_links.record(recorder, buffers, frame);
    }
}
