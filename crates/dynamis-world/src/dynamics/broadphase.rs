use super::FrameParams;
use super::stage::{CORE, RO, RW, Stage, UNIFORM, whole};
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
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.contacts.entries.cells)),
                    (RO, whole(&buffers.contacts.entries.colliders)),
                    (RW, buffers.counter(COUNTER_ENTRIES)),
                    (RW, whole(&buffers.contacts.pairs.major)),
                    (RW, whole(&buffers.contacts.pairs.minor)),
                    (RW, buffers.counter(COUNTER_PAIRS)),
                    (RW, buffers.counter(COUNTER_SPILLOVER_PAIRS)),
                    (RO, whole(&buffers.bodies.activity)),
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
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.contacts.large_bodies)),
                    (RW, buffers.counter(COUNTER_LARGE)),
                    (RW, whole(&buffers.contacts.pairs.major)),
                    (RW, whole(&buffers.contacts.pairs.minor)),
                    (RW, buffers.counter(COUNTER_PAIRS)),
                    (RO, whole(&buffers.bodies.colliders)),
                    (RW, buffers.counter(COUNTER_SPILLOVER_PAIRS)),
                    (RO, whole(&buffers.bodies.activity)),
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
        self.large_pairs.record(recorder, params.body_count);
    }
}
