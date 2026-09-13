use super::streams::{REACTION_WORDS, SoftDemand, SoftStream, SoftStreams};
use dynamis_engine::{MIN_SLOTS, Resources, STREAM_FLOOR, settled};
use dynamis_scene::Live;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SoftCapacity {
    pub particles: u32,
    pub links: u32,
    pub adjacency: u32,
}

pub fn capacity<R: Resources>(resources: &R) -> SoftCapacity {
    SoftCapacity {
        particles: resources.slots(SoftStream::Particles.into()),
        links: resources.slots(SoftStream::Links.into()),
        adjacency: resources.slots(SoftStream::Adjacency.into()),
    }
}

pub fn plan(live: &Live, idle: bool, bodies: u32, current: &SoftStreams) -> SoftDemand {
    SoftDemand {
        particles: settled(idle, current.particles.slots(), live.particles, MIN_SLOTS),
        links: settled(idle, current.links.slots(), live.links, MIN_SLOTS),
        adjacency: settled(
            idle,
            current.adjacency.slots(),
            live.adjacency,
            STREAM_FLOOR,
        ),
        bodies: settled(
            idle,
            current.reactions.slots() / REACTION_WORDS,
            bodies,
            MIN_SLOTS,
        ),
    }
}

pub const fn floor() -> SoftDemand {
    SoftDemand {
        particles: MIN_SLOTS,
        links: MIN_SLOTS,
        adjacency: STREAM_FLOOR,
        bodies: MIN_SLOTS,
    }
}
