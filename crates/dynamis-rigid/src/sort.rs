use dynamis_gpu::GpuSlot;
use dynamis_pass::{Resources, SlotRef};
use dynamis_sort::SortChannels;

use crate::RigidStream;

fn scratch<R: Resources>(resources: &R) -> [GpuSlot<'_>; 3] {
    [
        RigidStream::SortScratchMajor.whole().resolve(resources),
        RigidStream::SortScratchMinor.whole().resolve(resources),
        RigidStream::SortScratchPayload.whole().resolve(resources),
    ]
}

pub(crate) fn lanes<'a, R: Resources>(
    resources: &'a R,
    count: SlotRef,
    major: SlotRef,
    payload: SlotRef,
) -> SortChannels<'a> {
    let [scratch_major, scratch_minor, scratch_payload] = scratch(resources);
    SortChannels {
        generation: resources.generation(),
        count: count.resolve(resources),
        major: major.resolve(resources),
        minor: payload.resolve(resources),
        payload: payload.resolve(resources),
        scratch_major,
        scratch_minor,
        scratch_payload,
    }
}

pub(crate) fn keyed<'a, R: Resources>(
    resources: &'a R,
    count: SlotRef,
    major: SlotRef,
    minor: SlotRef,
    payload: SlotRef,
) -> SortChannels<'a> {
    let [scratch_major, scratch_minor, scratch_payload] = scratch(resources);
    SortChannels {
        generation: resources.generation(),
        count: count.resolve(resources),
        major: major.resolve(resources),
        minor: minor.resolve(resources),
        payload: payload.resolve(resources),
        scratch_major,
        scratch_minor,
        scratch_payload,
    }
}

pub(crate) fn lanes_dual<'a, R: Resources>(
    resources: &'a R,
    count: SlotRef,
    major: SlotRef,
    minor: SlotRef,
) -> SortChannels<'a> {
    let [scratch_major, scratch_minor, scratch_payload] = scratch(resources);
    SortChannels {
        generation: resources.generation(),
        count: count.resolve(resources),
        major: major.resolve(resources),
        minor: minor.resolve(resources),
        payload: RigidStream::SortDummyPayload.whole().resolve(resources),
        scratch_major,
        scratch_minor,
        scratch_payload,
    }
}
