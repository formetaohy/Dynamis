use super::FrameParams;
use super::stage::{CORE, Stage, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_ENTRIES, COUNTER_LARGE, COUNTER_PAIRS, COUNTER_SPILLOVER_PAIRS};
use dynamis_sort::RadixSort;

pub(super) struct Broadphase {
    broadphase_pairs: Stage,
    large_pairs: Stage,
}

impl Broadphase {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            broadphase_pairs: Stage::build(
                context,
                "broadphase_pairs",
                include_str!("shaders/broadphase_pairs.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("entry_cells", whole(&buffers.contacts.entries.cells)),
                    (
                        "entry_colliders",
                        whole(&buffers.contacts.entries.colliders),
                    ),
                    ("entry_count", buffers.counter(COUNTER_ENTRIES)),
                    ("pair_major", whole(&buffers.contacts.pairs.major)),
                    ("pair_minor", whole(&buffers.contacts.pairs.minor)),
                    ("pair_count", buffers.counter(COUNTER_PAIRS)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_PAIRS)),
                    ("body_activity", whole(&buffers.bodies.activity)),
                    ("collider_owners", whole(&buffers.bodies.collider_owners)),
                ],
                &[],
            ),
            large_pairs: Stage::build(
                context,
                "large_pairs",
                include_str!("shaders/large_pairs.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("large_bodies", whole(&buffers.contacts.large_bodies)),
                    ("large_count", buffers.counter(COUNTER_LARGE)),
                    ("pair_major", whole(&buffers.contacts.pairs.major)),
                    ("pair_minor", whole(&buffers.contacts.pairs.minor)),
                    ("pair_count", buffers.counter(COUNTER_PAIRS)),
                    ("colliders", whole(&buffers.bodies.colliders)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_PAIRS)),
                    ("body_activity", whole(&buffers.bodies.activity)),
                    ("collider_owners", whole(&buffers.bodies.collider_owners)),
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
            &buffers.contacts.entries.cells,
            &buffers.contacts.entries.colliders,
        );
        sort.sort(recorder, &channels, 4, 0, buffers.entry_capacity());
        self.broadphase_pairs
            .record_stride(recorder, buffers.entry_capacity());
        self.large_pairs.record(recorder, params.collider_count);
    }
}
