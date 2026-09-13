use crate::constant::{
    ELEMENT_AREA, ELEMENT_BEND, ELEMENT_DISTANCE, ELEMENT_PARTICLES, ELEMENT_VOLUME, NO_BODY,
    NO_SLOT,
};
use crate::{SoftElementRecord, SoftParticleRecord};
use dynamis_model::{SoftElement, SoftElementKind};

pub struct SoftParticleInit {
    pub position: [f32; 3],
    pub prev_position: [f32; 3],
    pub velocity: [f32; 3],
    pub radius: f32,
    pub inverse_mass: f32,
    pub friction: f32,
    pub neighbour_offset: u32,
    pub neighbour_count: u32,
    pub owner: u32,
    pub generation: u32,
}

impl SoftParticleRecord {
    pub fn build(init: SoftParticleInit) -> Self {
        Self {
            position: [
                init.position[0],
                init.position[1],
                init.position[2],
                init.radius,
            ],
            prev_position: [
                init.prev_position[0],
                init.prev_position[1],
                init.prev_position[2],
                init.inverse_mass,
            ],
            velocity: [
                init.velocity[0],
                init.velocity[1],
                init.velocity[2],
                init.friction,
            ],
            neighbour_offset: init.neighbour_offset,
            neighbour_count: init.neighbour_count,
            owner: init.owner,
            generation: init.generation,
        }
    }

    pub const fn cleared() -> Self {
        Self {
            position: [0.0; 4],
            prev_position: [0.0; 4],
            velocity: [0.0; 4],
            neighbour_offset: NO_SLOT,
            neighbour_count: 0,
            owner: NO_BODY,
            generation: 0,
        }
    }

    pub const fn radius(&self) -> f32 {
        self.position[3]
    }

    pub const fn inverse_mass(&self) -> f32 {
        self.prev_position[3]
    }

    pub const fn friction(&self) -> f32 {
        self.velocity[3]
    }
}

pub struct SoftElementInit {
    pub kind: u32,
    pub particles: [u32; ELEMENT_PARTICLES as usize],
    pub rest: f32,
    pub compliance: f32,
}

impl SoftElementRecord {
    pub fn build(init: SoftElementInit) -> Self {
        Self {
            particles: init.particles,
            rest: init.rest,
            compliance: init.compliance,
            lambda: 0.0,
            kind: init.kind,
        }
    }

    pub const fn cleared() -> Self {
        Self {
            particles: [NO_SLOT; ELEMENT_PARTICLES as usize],
            rest: 0.0,
            compliance: 0.0,
            lambda: 0.0,
            kind: ELEMENT_DISTANCE,
        }
    }
}

const _: () = assert!(ELEMENT_PARTICLES as usize == SoftElement::PARTICLES);
const _: () = assert!(SoftElement::UNUSED == NO_SLOT);
const _: () = assert!(SoftElementKind::Distance as u32 == ELEMENT_DISTANCE);
const _: () = assert!(SoftElementKind::Area as u32 == ELEMENT_AREA);
const _: () = assert!(SoftElementKind::Bend as u32 == ELEMENT_BEND);
const _: () = assert!(SoftElementKind::Volume as u32 == ELEMENT_VOLUME);
