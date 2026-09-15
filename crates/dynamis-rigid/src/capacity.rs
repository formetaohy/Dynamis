use super::streams::{RigidDemand, RigidStreams};
use dynamis_abi::{
    COUNTER_COLLIDERS, COUNTER_CONTACTS, COUNTER_EVENTS, COUNTER_IMPACTS, COUNTER_RESTING, Counters,
};
use dynamis_domain::{MIN_SLOTS, STREAM_FLOOR, grown, product, settled, unreported};

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
        events: streams.events.slots() / dynamis_gpu::SEGMENT_COUNT,
        impacts: streams.impacts.slots() / dynamis_gpu::SEGMENT_COUNT,
    }
}

pub fn plan(
    measured: &Counters,
    inputs: &RigidInputs,
    current: &RigidStreams,
    pairs: u32,
    release: bool,
) -> RigidDemand {
    let fresh = unreported(inputs.colliders, measured[COUNTER_COLLIDERS]);
    let frozen = measured[COUNTER_RESTING];
    let contacts = settled(
        current.contacts.slots(),
        product(fresh, FRESH_CONTACTS_PER_COLLIDER, "contact")
            .max(measured[COUNTER_CONTACTS])
            .max(frozen),
        STREAM_FLOOR,
        release,
    );
    let events = settled(
        current.events.slots() / dynamis_gpu::SEGMENT_COUNT,
        product(fresh, FRESH_EVENTS_PER_COLLIDER, "event")
            .max(measured[COUNTER_EVENTS])
            .max(frozen),
        STREAM_FLOOR,
        release,
    );
    let impacts = settled(
        current.impacts.slots() / dynamis_gpu::SEGMENT_COUNT,
        product(fresh, FRESH_IMPACTS_PER_COLLIDER, "impact").max(measured[COUNTER_IMPACTS]),
        STREAM_FLOOR,
        release,
    );
    let resting = current.resting_contacts.slots().max(frozen).max(contacts);
    let constraints = settled(
        current.constraint_rows.slots(),
        inputs.constraints,
        MIN_SLOTS,
        release,
    );
    RigidDemand {
        bodies: grown(current.body_activity.slots(), inputs.bodies, MIN_SLOTS),
        colliders: current
            .collider_aabbs
            .slots()
            .max(inputs.collider_pool)
            .max(MIN_SLOTS),
        constraints,
        pairs,
        contacts,
        resting,
        events,
        impacts,
        sort: RigidDemand::sort_slots(pairs, constraints).max(MIN_SLOTS),
        characters: settled(
            current.characters.slots(),
            inputs.characters,
            MIN_SLOTS,
            release,
        ),
        vehicles: settled(
            current.vehicles.slots(),
            inputs.vehicles,
            MIN_SLOTS,
            release,
        ),
    }
}

pub fn floor(pairs: u32) -> RigidDemand {
    RigidDemand {
        bodies: MIN_SLOTS,
        colliders: MIN_SLOTS,
        constraints: MIN_SLOTS,
        pairs,
        contacts: STREAM_FLOOR,
        resting: STREAM_FLOOR,
        events: STREAM_FLOOR,
        impacts: STREAM_FLOOR,
        sort: RigidDemand::sort_slots(pairs, MIN_SLOTS).max(MIN_SLOTS),
        characters: MIN_SLOTS,
        vehicles: MIN_SLOTS,
    }
}
