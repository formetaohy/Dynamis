use crate::dynamics::engine::streams;
use dynamis_gpu::Contents;
use dynamis_layout::{SoftLinkRecord, SoftParticleRecord};
use std::mem::size_of;

pub(crate) const DOMAIN: u32 = 2;

pub(crate) const REACTION_WORDS: u32 = 8;

streams! {
    SoftStreams, SoftStream, SoftDemand, DOMAIN, demand,
    demand {
        particles: u32,
        links: u32,
        adjacency: u32,
        bodies: u32,
    }
    streams {
        particles, Particles: "soft particles", size_of::<SoftParticleRecord>() as u64, Contents::Preserve, demand.particles;
        links, Links: "soft links", size_of::<SoftLinkRecord>() as u64, Contents::Preserve, demand.links;
        adjacency, Adjacency: "soft adjacency", 4, Contents::Preserve, demand.adjacency;
        link_deltas, LinkDeltas: "soft link deltas", 16, Contents::Reset, demand.links;
        reactions, Reactions: "soft reactions", 4, Contents::Preserve, demand.reaction_words();
    }
}

impl SoftDemand {
    pub(crate) const fn reaction_words(&self) -> u32 {
        self.bodies * REACTION_WORDS
    }
}
