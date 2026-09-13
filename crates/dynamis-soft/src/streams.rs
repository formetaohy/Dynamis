use dynamis_abi::{SoftElementRecord, SoftParticleRecord};
use dynamis_gpu::Contents;
use dynamis_pass::streams;
use std::mem::size_of;

pub const DOMAIN: u32 = 3;

pub const REACTION_WORDS: u32 = 8;

streams! {
    SoftStreams, SoftStream, SoftDemand, DOMAIN, demand,
    demand {
        particles: u32,
        elements: u32,
        adjacency: u32,
        bodies: u32,
    }
    streams {
        particles, Particles: "soft particles", size_of::<SoftParticleRecord>() as u64, Contents::Preserve, demand.particles;
        elements, Elements: "soft elements", size_of::<SoftElementRecord>() as u64, Contents::Preserve, demand.elements;
        adjacency, Adjacency: "soft adjacency", 4, Contents::Preserve, demand.adjacency;
        element_deltas, ElementDeltas: "soft element multipliers", 4, Contents::Reset, demand.elements;
        reactions, Reactions: "soft reactions", 4, Contents::Preserve, demand.reaction_words();
    }
}

impl SoftDemand {
    pub const fn reaction_words(&self) -> u32 {
        self.bodies * REACTION_WORDS
    }
}
