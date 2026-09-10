use crate::buffers::WorldBuffers;
use dynamis_gpu::{GpuBuffer, GpuSlot};
use dynamis_sort::SortChannels;

pub(super) fn lanes<'a>(
    buffers: &'a WorldBuffers,
    count: GpuSlot<'a>,
    keys_lo: &'a GpuBuffer,
    keys_hi: &'a GpuBuffer,
    values: &'a GpuBuffer,
) -> SortChannels<'a> {
    SortChannels {
        count,
        keys_lo,
        keys_hi,
        values,
        scratch_lo: &buffers.sort.scratch.keys_lo,
        scratch_hi: &buffers.sort.scratch.keys_hi,
        scratch_values: &buffers.sort.scratch.values,
    }
}
