use crate::SoftDomain;
use dynamis_abi::{ELEMENT_PARTICLES, SoftContactRecord, SoftElementRecord, SoftParticleRecord};
use dynamis_domain::Domain;
use dynamis_gpu::Contents;
use dynamis_pass::streams;
use std::mem::size_of;

pub const REACTION_WORDS: u32 = 8;

streams! {
    SoftStreams, SoftStream, SoftDemand, SoftDomain::ID, demand,
    demand {
        particles: u32,
        elements: u32,
        adjacency: u32,
        bodies: u32,
    }
    streams {
        particles, Particles: "soft particles", size_of::<SoftParticleRecord>() as u64, Contents::Durable, demand.particles;
        elements, Elements: "soft elements", size_of::<SoftElementRecord>() as u64, Contents::Durable, demand.elements;
        adjacency, Adjacency: "soft adjacency", 4, Contents::Durable, demand.adjacency;
        contributions, Contributions: "soft element contributions", size_of::<[f32; 4]>() as u64, Contents::Scratch, demand.element_contributions();
        contacts, Contacts: "soft particle contacts", size_of::<SoftContactRecord>() as u64, Contents::Scratch, demand.particles;
        pressure, Pressure: "soft particle pressure", size_of::<[f32; 4]>() as u64, Contents::Scratch, demand.particles;
        reactions, Reactions: "soft reactions", 4, Contents::Durable, demand.reaction_words();
    }
}

impl SoftDemand {
    pub const fn reaction_words(&self) -> u32 {
        self.bodies * REACTION_WORDS
    }

    pub const fn element_contributions(&self) -> u32 {
        self.elements * ELEMENT_PARTICLES
    }
}
