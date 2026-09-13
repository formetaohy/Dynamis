use super::streams::{BroadphaseDemand, BroadphaseStreams};
use dynamis_abi::{
    COUNTER_ENTRIES, COUNTER_PAIRS, COUNTER_SPILLOVER_ENTRIES, COUNTER_SPILLOVER_PAIRS, Counters,
    ENTRY_CELLS_PER_PARTICLE, MAX_CELLS_PER_COLLIDER,
};
use dynamis_engine::{MIN_SLOTS, STREAM_FLOOR, StreamWatch, product};
use dynamis_scene::Live;

const STREAM_DENSITY_PAIRS: u32 = 16;
const PLAN_COOLDOWN: u32 = 10;

pub struct Capacity {
    entries: StreamWatch,
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
            entries: StreamWatch::IDLE,
            pairs: StreamWatch::IDLE,
            cooldown: 0,
        }
    }

    pub fn plan(
        &mut self,
        measured: &Counters,
        live: &Live,
        current: &BroadphaseStreams,
    ) -> (BroadphaseDemand, bool) {
        let open = self.cooldown == 0;
        self.pairs.observe(
            measured[COUNTER_PAIRS],
            measured[COUNTER_SPILLOVER_PAIRS] > 0,
            current.pair_major.slots(),
        );
        self.entries.observe(
            measured[COUNTER_ENTRIES],
            measured[COUNTER_SPILLOVER_ENTRIES] > 0,
            current.entry_keys.slots(),
        );
        let pressured = self.pairs.pressured() || self.entries.pressured();
        let idle = open
            && !pressured
            && self.pairs.is_idle()
            && self.entries.is_idle()
            && live.body_commands == 0
            && live.constraint_commands == 0;
        if pressured {
            self.entries.settle(false);
            self.pairs.settle(false);
        } else if idle {
            self.entries.settle(true);
            self.pairs.settle(true);
            self.cooldown = PLAN_COOLDOWN;
        } else {
            self.cooldown = self.cooldown.saturating_sub(1);
        }
        let entry_budget = product(
            live.colliders,
            MAX_CELLS_PER_COLLIDER,
            "grid collider entry",
        )
        .checked_add(product(
            live.particles,
            ENTRY_CELLS_PER_PARTICLE,
            "grid particle entry",
        ))
        .unwrap_or_else(|| panic!("grid entry capacity exceeds the device index space"));
        let pair_budget = product(live.colliders, STREAM_DENSITY_PAIRS, "pair");
        let entries = if idle {
            self.entries
                .released(current.entry_keys.slots(), entry_budget)
        } else {
            self.entries
                .widened(current.entry_keys.slots(), entry_budget)
        };
        let pairs = if idle {
            self.pairs.released(current.pair_major.slots(), pair_budget)
        } else {
            self.pairs.widened(current.pair_major.slots(), pair_budget)
        };
        let sort = entries.max(pairs).max(MIN_SLOTS);
        (
            BroadphaseDemand {
                entries,
                pairs,
                sort,
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
