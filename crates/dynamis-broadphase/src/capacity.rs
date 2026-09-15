use super::streams::{BroadphaseDemand, BroadphaseStreams};
use dynamis_abi::{COUNTER_COLLIDERS, COUNTER_PAIRS, Counters, MAX_CELLS_PER_COLLIDER};
use dynamis_domain::{MIN_SLOTS, STREAM_FLOOR, product, settled, unreported};

const FRESH_PARTNERS_PER_COLLIDER: u32 = 16;

const _: () = assert!(
    FRESH_PARTNERS_PER_COLLIDER >= MAX_CELLS_PER_COLLIDER,
    "the freshly spawned pair reservation must cover every grid entry a collider can own"
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BroadphaseCapacity {
    pub entries: u32,
    pub pairs: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct BroadphaseInputs {
    pub colliders: u32,
    pub particles: u32,
}

pub fn plan(
    measured: &Counters,
    inputs: &BroadphaseInputs,
    current: &BroadphaseStreams,
    release: bool,
) -> BroadphaseDemand {
    let sources = inputs
        .colliders
        .checked_add(inputs.particles)
        .unwrap_or_else(|| panic!("grid entry sources exceed the device index space"));
    let entries = settled(
        current.entry_keys.slots(),
        product(sources, MAX_CELLS_PER_COLLIDER, "grid entry"),
        STREAM_FLOOR,
        release,
    );
    let fresh = unreported(inputs.colliders, measured[COUNTER_COLLIDERS]);
    let pairs = settled(
        current.pair_major.slots(),
        product(fresh, FRESH_PARTNERS_PER_COLLIDER, "pair").max(measured[COUNTER_PAIRS]),
        STREAM_FLOOR,
        release,
    );
    BroadphaseDemand {
        entries,
        pairs,
        sort: entries.max(pairs).max(MIN_SLOTS),
    }
}

pub fn floor() -> BroadphaseDemand {
    BroadphaseDemand {
        entries: STREAM_FLOOR,
        pairs: STREAM_FLOOR,
        sort: STREAM_FLOOR,
    }
}
