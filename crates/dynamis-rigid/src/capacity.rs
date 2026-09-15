use super::streams::{RigidDemand, RigidStreams};
use dynamis_abi::{
    COUNTER_COLLIDERS, COUNTER_CONTACTS, COUNTER_EVENTS, COUNTER_IMPACTS, COUNTER_REFUSED_CONTACTS,
    COUNTER_REFUSED_EVENTS, COUNTER_REFUSED_IMPACTS, COUNTER_RESTING, Counters,
};
use dynamis_domain::{MIN_SLOTS, StreamWatch, product, settled, unreported};

const FRESH_EVENTS_PER_COLLIDER: u32 = 8;
const FRESH_CONTACTS_PER_COLLIDER: u32 = 4;
const FRESH_IMPACTS_PER_COLLIDER: u32 = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RigidCapacity {
    pub pairs: u32,
    pub contacts: u32,
    pub resting: u32,
    pub events: u32,
    pub impacts: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct RigidInputs {
    pub bodies: u32,
    pub colliders: u32,
    pub collider_pool: u32,
    pub constraints: u32,
    pub queries: u32,
    pub observed: u32,
    pub observed_joints: u32,
    pub ccd: bool,
    pub impacts: bool,
    pub characters: u32,
    pub vehicles: u32,
}

pub fn capacity(streams: &RigidStreams) -> RigidCapacity {
    RigidCapacity {
        pairs: streams.contact_valid.slots(),
        contacts: streams.contacts.slots(),
        resting: streams.resting_contacts.slots(),
        events: streams.events.slots() / dynamis_gpu::EVENT_SLOTS,
        impacts: streams.impacts.slots() / dynamis_gpu::EVENT_SLOTS,
    }
}

pub struct Capacity {
    events: StreamWatch,
    impacts: StreamWatch,
    contacts: StreamWatch,
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
            impacts: StreamWatch::IDLE,
            contacts: StreamWatch::IDLE,
        }
    }

    pub fn plan(
        &mut self,
        measured: &Counters,
        inputs: &RigidInputs,
        idle: bool,
        pairs: u32,
        current: &RigidStreams,
    ) -> RigidDemand {
        self.events.observe(
            measured[COUNTER_EVENTS],
            measured[COUNTER_REFUSED_EVENTS] > 0,
            current.events.slots() / dynamis_gpu::EVENT_SLOTS,
        );
        if self.events.pressured() {
            self.events.settle(false);
        } else if idle {
            self.events.settle(true);
        }
        let thawing = measured[COUNTER_RESTING];
        let fresh = unreported(inputs.colliders, measured[COUNTER_COLLIDERS]);
        let event_budget = product(fresh, FRESH_EVENTS_PER_COLLIDER, "event").max(thawing);
        let events = if idle {
            self.events.released(
                current.events.slots() / dynamis_gpu::EVENT_SLOTS,
                event_budget,
            )
        } else {
            self.events.widened(
                current.events.slots() / dynamis_gpu::EVENT_SLOTS,
                event_budget,
            )
        };
        self.impacts.observe(
            measured[COUNTER_IMPACTS],
            measured[COUNTER_REFUSED_IMPACTS] > 0,
            current.impacts.slots() / dynamis_gpu::EVENT_SLOTS,
        );
        if self.impacts.pressured() {
            self.impacts.settle(false);
        } else if idle {
            self.impacts.settle(true);
        }
        let impact_budget = product(fresh, FRESH_IMPACTS_PER_COLLIDER, "impact");
        let impacts = if idle {
            self.impacts.released(
                current.impacts.slots() / dynamis_gpu::EVENT_SLOTS,
                impact_budget,
            )
        } else {
            self.impacts.widened(
                current.impacts.slots() / dynamis_gpu::EVENT_SLOTS,
                impact_budget,
            )
        };
        self.contacts.observe(
            measured[COUNTER_CONTACTS],
            measured[COUNTER_REFUSED_CONTACTS] > 0,
            current.contacts.slots(),
        );
        if self.contacts.pressured() {
            self.contacts.settle(false);
        } else if idle {
            self.contacts.settle(true);
        }
        let contact_budget = product(fresh, FRESH_CONTACTS_PER_COLLIDER, "contact").max(thawing);
        let contacts = if idle {
            self.contacts
                .released(current.contacts.slots(), contact_budget)
        } else {
            self.contacts
                .doubled(current.contacts.slots(), contact_budget)
        };
        let resting = current
            .resting_contacts
            .slots()
            .max(measured[COUNTER_RESTING])
            .max(contacts);
        let bodies = dynamis_domain::grown(current.body_activity.slots(), inputs.bodies, MIN_SLOTS);
        let colliders = current
            .collider_aabbs
            .slots()
            .max(inputs.collider_pool)
            .max(MIN_SLOTS);
        let constraints = settled(
            idle,
            current.constraint_rows.slots(),
            inputs.constraints,
            MIN_SLOTS,
        );
        let sort = RigidDemand::sort_slots(pairs, constraints).max(MIN_SLOTS);
        let characters = settled(
            idle,
            current.characters.slots(),
            inputs.characters,
            MIN_SLOTS,
        );
        let vehicles = settled(idle, current.vehicles.slots(), inputs.vehicles, MIN_SLOTS);
        RigidDemand {
            bodies,
            colliders,
            constraints,
            pairs,
            contacts,
            resting,
            events,
            impacts,
            sort,
            characters,
            vehicles,
        }
    }

    pub fn floor(pairs: u32) -> RigidDemand {
        RigidDemand {
            bodies: MIN_SLOTS,
            colliders: MIN_SLOTS,
            constraints: MIN_SLOTS,
            pairs,
            contacts: dynamis_domain::STREAM_FLOOR,
            resting: dynamis_domain::STREAM_FLOOR,
            events: dynamis_domain::STREAM_FLOOR,
            impacts: dynamis_domain::STREAM_FLOOR,
            sort: RigidDemand::sort_slots(pairs, MIN_SLOTS).max(MIN_SLOTS),
            characters: MIN_SLOTS,
            vehicles: MIN_SLOTS,
        }
    }
}
