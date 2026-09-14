use super::streams::{BroadphaseDemand, BroadphaseStreams};
use dynamis_abi::{COUNTER_PAIRS, COUNTER_REFUSED_PAIRS, Counters, MAX_CELLS_PER_COLLIDER};
use dynamis_domain::{MIN_SLOTS, STREAM_FLOOR, StreamWatch, product, settled};

const PARTNERS_PER_COLLIDER: u32 = 16;
const PLAN_COOLDOWN: u32 = 10;

const _: () = assert!(
    PARTNERS_PER_COLLIDER >= MAX_CELLS_PER_COLLIDER,
    "the candidate pair floor must cover every grid entry a collider can own"
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
    pub pending_commands: bool,
}

pub struct Capacity {
    pairs: StreamWatch,
    cooldown: u32,
}

impl Default for Capacity {
    fn default() -> Self {
        Self::new()
    }
}

impl Capacity {
    pub const fn new() -> Self {
        Self {
            pairs: StreamWatch::IDLE,
            cooldown: 0,
        }
    }

    pub fn plan(
        &mut self,
        measured: &Counters,
        inputs: &BroadphaseInputs,
        current: &BroadphaseStreams,
    ) -> (BroadphaseDemand, bool) {
        let open = self.cooldown == 0;
        self.pairs.observe(
            measured[COUNTER_PAIRS],
            measured[COUNTER_REFUSED_PAIRS] > 0,
            current.pair_major.slots(),
        );
        let pressured = self.pairs.pressured();
        let idle = open && !pressured && self.pairs.is_idle() && !inputs.pending_commands;
        if pressured {
            self.pairs.settle(false);
        } else if idle {
            self.pairs.settle(true);
            self.cooldown = PLAN_COOLDOWN;
        } else {
            self.cooldown = self.cooldown.saturating_sub(1);
        }
        let sources = inputs
            .colliders
            .checked_add(inputs.particles)
            .unwrap_or_else(|| panic!("grid entry sources exceed the device index space"));
        let entries = settled(
            idle,
            current.entry_keys.slots(),
            product(sources, MAX_CELLS_PER_COLLIDER, "grid entry"),
            STREAM_FLOOR,
        );
        let pair_floor = product(inputs.colliders, PARTNERS_PER_COLLIDER, "pair");
        let pairs = if idle {
            self.pairs.released(current.pair_major.slots(), pair_floor)
        } else {
            self.pairs.widened(current.pair_major.slots(), pair_floor)
        };
        (
            BroadphaseDemand {
                entries,
                pairs,
                sort: entries.max(pairs).max(MIN_SLOTS),
            },
            idle,
        )
    }

    pub fn floor() -> BroadphaseDemand {
        BroadphaseDemand {
            entries: STREAM_FLOOR,
            pairs: STREAM_FLOOR,
            sort: STREAM_FLOOR,
        }
    }
}
