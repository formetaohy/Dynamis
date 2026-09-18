use super::streams::{BroadphaseDemand, BroadphaseStreams};
use dynamis_abi::{COUNTER_MOVABLE_COLLIDERS, COUNTER_PAIRS, Counters, MAX_CELLS_PER_COLLIDER};
use dynamis_domain::{MIN_SLOTS, STREAM_FLOOR, grown, product, settled, unreported};

const FRESH_PARTNERS_PER_MOVABLE_COLLIDER: u32 = 16;

const _: () = assert!(
    FRESH_PARTNERS_PER_MOVABLE_COLLIDER >= MAX_CELLS_PER_COLLIDER,
    "the freshly spawned pair reservation must cover every grid entry a movable collider can own"
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BroadphaseCapacity {
    pub entries: u32,
    pub pairs: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct BroadphaseInputs {
    pub bodies: u32,
    pub colliders: u32,
    pub movable_colliders: u32,
    pub immovable_colliders: u32,
    pub particles: u32,
    pub entry_base: u32,
    pub moving_slots: u32,
    pub immovable_rebuild: bool,
    pub resting_rebuild: bool,
}

pub fn plan(
    measured: &Counters,
    inputs: &BroadphaseInputs,
    current: &BroadphaseStreams,
    immovable: u32,
    release: bool,
) -> BroadphaseDemand {
    let movable_sources = inputs
        .movable_colliders
        .checked_add(inputs.particles)
        .unwrap_or_else(|| panic!("movable grid entry sources exceed the device index space"));
    let immovable_entries = settled(
        immovable,
        product(
            inputs.immovable_colliders,
            MAX_CELLS_PER_COLLIDER,
            "immovable grid entry",
        ),
        STREAM_FLOOR,
        release,
    );
    let moving_entries = settled(
        current.entry_keys.slots().saturating_sub(immovable),
        product(
            movable_sources,
            MAX_CELLS_PER_COLLIDER,
            "movable grid entry",
        ),
        STREAM_FLOOR,
        release,
    );
    let entries = immovable_entries + moving_entries;
    let fresh = unreported(
        inputs.movable_colliders,
        measured[COUNTER_MOVABLE_COLLIDERS],
    );
    let pairs = settled(
        current.pair_major.slots(),
        product(fresh, FRESH_PARTNERS_PER_MOVABLE_COLLIDER, "pair").max(measured[COUNTER_PAIRS]),
        STREAM_FLOOR,
        release,
    );
    BroadphaseDemand {
        immovable: immovable_entries,
        entries,
        pairs,
        sort: entries.max(pairs).max(MIN_SLOTS),
        bodies: grown(current.body_admitted.slots(), inputs.bodies, MIN_SLOTS),
    }
}

pub fn floor() -> BroadphaseDemand {
    let immovable = STREAM_FLOOR;
    let entries = immovable + STREAM_FLOOR;
    let pairs = STREAM_FLOOR;
    BroadphaseDemand {
        immovable,
        entries,
        pairs,
        sort: entries.max(pairs).max(MIN_SLOTS),
        bodies: MIN_SLOTS,
    }
}
