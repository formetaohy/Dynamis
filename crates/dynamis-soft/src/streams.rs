use crate::SoftDomain;
use dynamis_abi::{
    ELEMENT_PARTICLES, SoftAttachmentRecord, SoftBodyRecord, SoftContactRecord, SoftElementRecord,
    SoftParticleRecord,
};
use dynamis_domain::Domain;
use dynamis_gpu::Contents;
use dynamis_pass::streams;

pub const REACTION_WORDS: u32 = 8;

streams! {
    SoftStreams, SoftStream, SoftDemand, SoftDomain::ID, demand,
    demand {
        particles: u32,
        elements: u32,
        attachments: u32,
        adjacency: u32,
        soft_bodies: u32,
        rigid_bodies: u32,
    }
    streams {
        particles, Particles: "soft particles", SoftParticleRecord, 1, Contents::Durable, demand.particles;
        elements, Elements: "soft elements", SoftElementRecord, 1, Contents::Durable, demand.elements;
        attachments, Attachments: "soft attachments", SoftAttachmentRecord, 1, Contents::Durable, demand.attachments;
        adjacency, Adjacency: "soft adjacency", u32, 1, Contents::Durable, demand.adjacency;
        bodies, BodyStates: "soft body states", SoftBodyRecord, 1, Contents::Durable, demand.soft_bodies;
        contributions, Contributions: "soft element contributions", [f32; 4], 1, Contents::Scratch, demand.element_contributions();
        contacts, Contacts: "soft particle contacts", SoftContactRecord, 1, Contents::Scratch, demand.particles;
        pressure, Pressure: "soft particle pressure", [f32; 4], 1, Contents::Scratch, demand.particles;
        reactions, Reactions: "soft reactions", u32, 1, Contents::Durable, demand.reaction_words();
    }
}

impl SoftDemand {
    pub const fn reaction_words(&self) -> u32 {
        self.rigid_bodies * REACTION_WORDS
    }

    pub const fn element_contributions(&self) -> u32 {
        self.elements * ELEMENT_PARTICLES
    }
}
