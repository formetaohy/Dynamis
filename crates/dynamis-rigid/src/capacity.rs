use super::streams::{RigidDemand, RigidStreams};
use dynamis_abi::{COUNTER_EVENTS, COUNTER_SPILLOVER_EVENTS, Counters};
use dynamis_pass::{MIN_SLOTS, StreamWatch, settled};
use dynamis_state::Live;

const STREAM_DENSITY_EVENTS: u32 = 8;

pub struct Capacity {
    events: StreamWatch,
}

impl Default for Capacity {
    fn default() -> Self {
        Self::new()
    }
}

impl Capacity {
    pub const fn new() -> Self {
        Self {
            events: StreamWatch::IDLE,
        }
    }

    pub fn plan(
        &mut self,
        measured: &Counters,
        live: &Live,
        idle: bool,
        pairs: u32,
        current: &RigidStreams,
    ) -> RigidDemand {
        self.events.observe(
            measured[COUNTER_EVENTS],
            measured[COUNTER_SPILLOVER_EVENTS] > 0,
            current.events.slots() / dynamis_pass::EVENT_SLOTS,
        );
        if self.events.pressured() {
            self.events.settle(false);
        } else if idle {
            self.events.settle(true);
        }
        let event_budget = dynamis_pass::product(live.colliders, STREAM_DENSITY_EVENTS, "event");
        let events = if idle {
            self.events.released(
                current.events.slots() / dynamis_pass::EVENT_SLOTS,
                event_budget,
            )
        } else {
            self.events.widened(
                current.events.slots() / dynamis_pass::EVENT_SLOTS,
                event_budget,
            )
        };
        let bodies = dynamis_pass::grown(current.body_activity.slots(), live.bodies, MIN_SLOTS);
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
        let sort = RigidDemand::sort_slots(pairs, constraints).max(MIN_SLOTS);
        RigidDemand {
            bodies,
            colliders,
            constraints,
            pairs,
            events,
            sort,
        }
    }

    pub fn floor(pairs: u32) -> RigidDemand {
        RigidDemand {
            bodies: MIN_SLOTS,
            colliders: MIN_SLOTS,
            constraints: MIN_SLOTS,
            pairs,
            events: dynamis_pass::STREAM_FLOOR,
            sort: RigidDemand::sort_slots(pairs, MIN_SLOTS).max(MIN_SLOTS),
        }
    }
}
