use super::streams::{REACTION_WORDS, SoftDemand, SoftStreams};
use dynamis_domain::{MIN_SLOTS, STREAM_FLOOR, settled};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SoftCapacity {
    pub particles: u32,
    pub elements: u32,
    pub attachments: u32,
    pub adjacency: u32,
    pub bodies: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SoftInputs {
    pub particles: u32,
    pub elements: u32,
    pub attachments: u32,
    pub adjacency: u32,
    pub bodies: u32,
    pub material: bool,
}

pub fn capacity(streams: &SoftStreams) -> SoftCapacity {
    SoftCapacity {
        particles: streams.particles.slots(),
        elements: streams.elements.slots(),
        attachments: streams.attachments.slots(),
        adjacency: streams.adjacency.slots(),
        bodies: streams.bodies.slots(),
    }
}

pub fn plan(
    inputs: &SoftInputs,
    idle: bool,
    rigid_bodies: u32,
    current: &SoftStreams,
) -> SoftDemand {
    SoftDemand {
        particles: settled(idle, current.particles.slots(), inputs.particles, MIN_SLOTS),
        elements: settled(idle, current.elements.slots(), inputs.elements, MIN_SLOTS),
        attachments: settled(
            idle,
            current.attachments.slots(),
            inputs.attachments,
            MIN_SLOTS,
        ),
        adjacency: settled(
            idle,
            current.adjacency.slots(),
            inputs.adjacency,
            STREAM_FLOOR,
        ),
        soft_bodies: settled(idle, current.bodies.slots(), inputs.bodies, MIN_SLOTS),
        rigid_bodies: settled(
            idle,
            current.reactions.slots() / REACTION_WORDS,
            rigid_bodies,
            MIN_SLOTS,
        ),
    }
}

pub const fn floor() -> SoftDemand {
    SoftDemand {
        particles: MIN_SLOTS,
        elements: MIN_SLOTS,
        attachments: MIN_SLOTS,
        adjacency: STREAM_FLOOR,
        soft_bodies: MIN_SLOTS,
        rigid_bodies: MIN_SLOTS,
    }
}
