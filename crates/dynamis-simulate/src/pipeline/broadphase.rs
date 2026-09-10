use super::FrameParams;
use super::dispatch::{BROADPHASE_PAIRS, SORT_ENTRIES};
use super::sort;
use super::stage::{RO, RW, Stage, UNIFORM, whole};
use crate::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_ENTRIES, COUNTER_LARGE, COUNTER_PAIRS, COUNTER_SPILLOVER_PAIRS};
use dynamis_sort::{RadixSort, key_words};

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
                include_str!("../shaders/broadphase_pairs.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.contacts.entries.keys_hi)),
                    (RO, whole(&buffers.contacts.entries.keys_lo)),
                    (RW, buffers.counter(COUNTER_ENTRIES)),
                    (RW, whole(&buffers.contacts.pairs.keys_hi)),
                    (RW, whole(&buffers.contacts.pairs.keys_lo)),
                    (RW, buffers.counter(COUNTER_PAIRS)),
                    (RW, buffers.counter(COUNTER_SPILLOVER_PAIRS)),
                ],
                &[],
            ),
            large_pairs: Stage::build(
                context,
                "large_pairs",
                include_str!("../shaders/large_pairs.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.contacts.large_bodies)),
                    (RW, buffers.counter(COUNTER_LARGE)),
                    (RW, whole(&buffers.contacts.pairs.keys_hi)),
                    (RW, whole(&buffers.contacts.pairs.keys_lo)),
                    (RW, buffers.counter(COUNTER_PAIRS)),
                    (RO, whole(&buffers.bodies.colliders)),
                    (RW, buffers.counter(COUNTER_SPILLOVER_PAIRS)),
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
        let words = key_words(buffers.collider_rows());
        let channels = sort::lanes(
            buffers,
            buffers.counter(COUNTER_ENTRIES),
            &buffers.contacts.entries.keys_lo,
            &buffers.contacts.entries.keys_hi,
            &buffers.sort.values,
        );
        sort.sort(
            recorder,
            &channels,
            words,
            4,
            &buffers.dispatch,
            SORT_ENTRIES,
        );
        self.broadphase_pairs
            .record_indirect(recorder, &buffers.dispatch, BROADPHASE_PAIRS);
        self.large_pairs.record(recorder, params.body_count);
    }
}
