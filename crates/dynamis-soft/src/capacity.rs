use super::streams::{SoftDemand, SoftStreams};
use dynamis_domain::{MIN_SLOTS, STREAM_FLOOR, settled};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SoftCapacity {
    pub particles: u32,
    pub elements: u32,
    pub attachments: u32,
    pub adjacency: u32,
    pub bodies: u32,
    pub edits: u32,
    pub body_edits: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SoftInputs {
    pub particles: u32,
    pub elements: u32,
    pub attachments: u32,
    pub adjacency: u32,
    pub bodies: u32,
    pub edits: u32,
    pub body_edits: u32,
    pub material: bool,
}

pub fn capacity(streams: &SoftStreams) -> SoftCapacity {
    SoftCapacity {
        particles: streams.particles.slots(),
        elements: streams.elements.slots(),
        attachments: streams.attachments.slots(),
        adjacency: streams.adjacency.slots(),
        bodies: streams.bodies.slots(),
        edits: streams.edits.slots(),
        body_edits: streams.body_edits.slots(),
    }
}

pub fn plan(inputs: &SoftInputs, current: &SoftStreams, release: bool) -> SoftDemand {
    SoftDemand {
        particles: settled(
            current.particles.slots(),
            inputs.particles,
            MIN_SLOTS,
            release,
        ),
        elements: settled(
            current.elements.slots(),
            inputs.elements,
            MIN_SLOTS,
            release,
        ),
        attachments: settled(
            current.attachments.slots(),
            inputs.attachments,
            MIN_SLOTS,
            release,
        ),
        adjacency: settled(
            current.adjacency.slots(),
            inputs.adjacency,
            STREAM_FLOOR,
            release,
        ),
        soft_bodies: settled(current.bodies.slots(), inputs.bodies, MIN_SLOTS, release),
        edits: settled(current.edits.slots(), inputs.edits, MIN_SLOTS, release),
        body_edits: settled(
            current.body_edits.slots(),
            inputs.body_edits,
            MIN_SLOTS,
            release,
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
        edits: MIN_SLOTS,
        body_edits: MIN_SLOTS,
    }
}
