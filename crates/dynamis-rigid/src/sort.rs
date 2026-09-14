use dynamis_gpu::TypedSlot;
use dynamis_gpu::{Resources, SlotRef};
use dynamis_sort::SortChannels;

use crate::RigidStream;

fn scratch<R: Resources>(resources: &R) -> [TypedSlot<'_>; 3] {
    [
        RigidStream::SortScratchMajor.whole().resolve(resources),
        RigidStream::SortScratchMinor.whole().resolve(resources),
        RigidStream::SortScratchPayload.whole().resolve(resources),
    ]
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
        count: count.resolve(resources),
        major: major.resolve(resources),
        minor: minor.resolve(resources),
        payload: RigidStream::SortDummyPayload.whole().resolve(resources),
        scratch_major,
        scratch_minor,
        scratch_payload,
    }
}
