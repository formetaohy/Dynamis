use super::streams::{RigidDemand, RigidStreams};
use crate::dynamics::capacity::{
    Live, MIN_SLOTS, STREAM_FLOOR, StreamWatch, grown, product, settled,
};
use dynamis_layout::{
    COUNTER_ENTRIES, COUNTER_EVENTS, COUNTER_PAIRS, COUNTER_SPILLOVER_ENTRIES,
    COUNTER_SPILLOVER_EVENTS, COUNTER_SPILLOVER_PAIRS, COUNTER_SPILLOVER_RESTING, Counters,
    MAX_CELLS_PER_COLLIDER,
};

const STREAM_DENSITY_PAIRS: u32 = 16;
const STREAM_DENSITY_EVENTS: u32 = 8;
const PLAN_COOLDOWN: u32 = 10;

pub(crate) struct Capacity {
    entries: StreamWatch,
    pairs: StreamWatch,
    events: StreamWatch,
    cooldown: u32,
}

impl Capacity {
    pub(crate) const fn new() -> Self {
        Self {
            entries: StreamWatch::IDLE,
            pairs: StreamWatch::IDLE,
            events: StreamWatch::IDLE,
            cooldown: 0,
        }
    }

    pub(crate) fn plan(
        &mut self,
        measured: &Counters,
        live: &Live,
        current: &RigidStreams,
    ) -> (RigidDemand, bool) {
        self.cooldown = self.cooldown.saturating_sub(1);
        let open = self.cooldown == 0;
        self.pairs.observe(
            measured[COUNTER_PAIRS],
            measured[COUNTER_SPILLOVER_PAIRS] > 0 || measured[COUNTER_SPILLOVER_RESTING] > 0,
            current.pair_capacity(),
            open,
        );
        self.entries.observe(
            measured[COUNTER_ENTRIES],
            measured[COUNTER_SPILLOVER_ENTRIES] > 0,
            current.entry_capacity(),
            open,
        );
        self.events.observe(
            measured[COUNTER_EVENTS],
            measured[COUNTER_SPILLOVER_EVENTS] > 0,
            current.event_capacity(),
            open,
        );
        let pressured =
            self.pairs.pressured() || self.entries.pressured() || self.events.pressured();
        let idle = open
            && !pressured
            && self.pairs.is_idle()
            && self.entries.is_idle()
            && self.events.is_idle()
            && live.body_commands == 0
            && live.constraint_commands == 0;
        if open && (pressured || idle) {
            self.cooldown = PLAN_COOLDOWN;
            self.entries.settle(idle);
            self.pairs.settle(idle);
            self.events.settle(idle);
        }
        let entry_budget = product(live.colliders, MAX_CELLS_PER_COLLIDER, "grid entry");
        let pair_budget = product(live.colliders, STREAM_DENSITY_PAIRS, "pair");
        let event_budget = product(live.colliders, STREAM_DENSITY_EVENTS, "event");
        let entries = if idle {
            self.entries
                .released(current.entry_capacity(), entry_budget)
        } else {
            self.entries.widened(current.entry_capacity(), entry_budget)
        };
        let pairs = if idle {
            self.pairs.released(current.pair_capacity(), pair_budget)
        } else {
            self.pairs.widened(current.pair_capacity(), pair_budget)
        };
        let events = if idle {
            self.events.released(current.event_capacity(), event_budget)
        } else {
            self.events.widened(current.event_capacity(), event_budget)
        };
        let bodies = grown(current.body_activity.slots(), live.bodies, MIN_SLOTS);
        let colliders = current
            .collider_aabbs
            .slots()
            .max(live.collider_pool)
            .max(MIN_SLOTS);
        let constraints = settled(
            idle,
            current.constraint_rows.slots(),
            live.constraints,
            MIN_SLOTS,
        );
        let sort = RigidDemand::sort_slots(entries, pairs, constraints);
        (
            RigidDemand {
                bodies,
                colliders,
                constraints,
                entries,
                pairs,
                events,
                sort,
            },
            idle,
        )
    }

    pub(crate) fn floor() -> RigidDemand {
        RigidDemand {
            bodies: MIN_SLOTS,
            colliders: MIN_SLOTS,
            constraints: MIN_SLOTS,
            entries: STREAM_FLOOR,
            pairs: STREAM_FLOOR,
            events: STREAM_FLOOR,
            sort: RigidDemand::sort_slots(STREAM_FLOOR, STREAM_FLOOR, MIN_SLOTS),
        }
    }
}
