use super::streams::{REACTION_WORDS, SoftDemand, SoftStreams};
use crate::dynamics::capacity::{Live, MIN_SLOTS, STREAM_FLOOR, settled};

pub(crate) fn plan(live: &Live, idle: bool, bodies: u32, current: &SoftStreams) -> SoftDemand {
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

pub(crate) const fn floor() -> SoftDemand {
    SoftDemand {
        particles: MIN_SLOTS,
        links: MIN_SLOTS,
        adjacency: STREAM_FLOOR,
        bodies: MIN_SLOTS,
    }
}
