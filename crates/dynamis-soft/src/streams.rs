use crate::SoftDomain;
use dynamis_abi::{
    ContactEventRecord, ELEMENT_PARTICLES, SoftAttachmentRecord, SoftBodyEditRecord,
    SoftBodyRecord, SoftContactRecord, SoftEditRecord, SoftElementRecord, SoftParticleRecord,
};
use dynamis_domain::Domain;
use dynamis_domain::StreamWriters;
use dynamis_domain::streams;
use dynamis_gpu::Retention;
use dynamis_gpu::SEGMENT_COUNT;

streams! {
    SoftStreams, SoftStream, SoftDemand, SoftDomain::ID, demand,
    demand {
        particles: u32,
        elements: u32,
        attachments: u32,
        adjacency: u32,
        soft_bodies: u32,
        edits: u32,
        body_edits: u32,
        events: u32,
    }
    streams {
        particles, Particles: "soft particles", SoftParticleRecord, 1, Retention::Durable, StreamWriters::HostThenDevice, demand.particles;
        elements, Elements: "soft elements", SoftElementRecord, 1, Retention::Durable, StreamWriters::HostThenDevice, demand.elements;
        attachments, Attachments: "soft attachments", SoftAttachmentRecord, 1, Retention::Durable, StreamWriters::Host, demand.attachments;
        adjacency, Adjacency: "soft adjacency", u32, 1, Retention::Durable, StreamWriters::Host, demand.adjacency;
        bodies, BodyStates: "soft body states", SoftBodyRecord, 1, Retention::Durable, StreamWriters::HostThenDevice, demand.soft_bodies;
        edits, Edits: "soft particle edits", SoftEditRecord, 1, Retention::Scratch, StreamWriters::Host, demand.edits;
        body_edits, BodyEdits: "soft body edits", SoftBodyEditRecord, 1, Retention::Scratch, StreamWriters::Host, demand.body_edits;
        contributions, Contributions: "soft element contributions", [f32; 4], 1, Retention::Scratch, StreamWriters::Device, demand.element_contributions();
        contacts, Contacts: "soft particle contacts", SoftContactRecord, 1, Retention::Scratch, StreamWriters::Device, demand.particles;
        pressure, Pressure: "soft particle pressure", [f32; 4], 1, Retention::Scratch, StreamWriters::Device, demand.particles;
        events, Events: "soft contact events", ContactEventRecord, 1, Retention::Scratch, StreamWriters::Device, demand.events.saturating_mul(SEGMENT_COUNT);
    }
}

impl SoftDemand {
    pub const fn element_contributions(&self) -> u32 {
        self.elements * ELEMENT_PARTICLES
    }
}
