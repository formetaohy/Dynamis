use super::FrameParams;
use super::stage::{GRID_INDEX, Stage, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{
    COUNTER_COARSE_ACTIVE, COUNTER_ENTRIES, COUNTER_GRID_LEVELS, COUNTER_PAIRS,
    COUNTER_SPILLOVER_PAIRS,
};
use dynamis_sort::RadixSort;

pub(super) struct Broadphase {
    cell_pairs: Stage,
    level_links: Stage,
}

impl Broadphase {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            cell_pairs: Stage::build(
                context,
                "cell_pairs",
                include_str!("shaders/cell_pairs.wgsl"),
                per_row,
                GRID_INDEX,
                &[
                    ("params", whole(&buffers.params)),
                    ("entry_keys", whole(&buffers.grid_entry_keys)),
                    ("entry_colliders", whole(&buffers.grid_entry_colliders)),
                    ("entry_count", buffers.counter(COUNTER_ENTRIES)),
                    ("pair_major", whole(&buffers.pair_major)),
                    ("pair_minor", whole(&buffers.pair_minor)),
                    ("pair_count", buffers.counter(COUNTER_PAIRS)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_PAIRS)),
                    ("body_activity", whole(&buffers.body_activity)),
                    ("collider_owners", whole(&buffers.collider_owners)),
                ],
                &[],
            ),
            level_links: Stage::build(
                context,
                "level_links",
                include_str!("shaders/level_links.wgsl"),
                per_row,
                GRID_INDEX,
                &[
                    ("params", whole(&buffers.params)),
                    ("entry_keys", whole(&buffers.grid_entry_keys)),
                    ("entry_colliders", whole(&buffers.grid_entry_colliders)),
                    ("entry_count", buffers.counter(COUNTER_ENTRIES)),
                    ("pair_major", whole(&buffers.pair_major)),
                    ("pair_minor", whole(&buffers.pair_minor)),
                    ("pair_count", buffers.counter(COUNTER_PAIRS)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_PAIRS)),
                    ("aabbs", whole(&buffers.collider_aabbs)),
                    ("collider_owners", whole(&buffers.collider_owners)),
                    ("body_activity", whole(&buffers.body_activity)),
                    ("levels", buffers.counter(COUNTER_GRID_LEVELS)),
                    ("active_coarse", buffers.counter(COUNTER_COARSE_ACTIVE)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        params: &FrameParams,
        sort: &RadixSort,
    ) {
        let channels = buffers.sort_lanes(
            buffers.counter(COUNTER_ENTRIES),
            &buffers.grid_entry_keys,
            &buffers.grid_entry_colliders,
        );
        sort.sort(recorder, &channels, 4, 0, buffers.entry_capacity());
        self.cell_pairs
            .record_stride(recorder, buffers.entry_capacity());
        self.level_links.record(recorder, params.collider_count);
    }
}
