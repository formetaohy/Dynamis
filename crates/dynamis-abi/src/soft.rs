use crate::constant::{
    ELEMENT_AREA, ELEMENT_BEND, ELEMENT_BROKEN, ELEMENT_DISTANCE, ELEMENT_KIND_MASK,
    ELEMENT_PARTICLES, ELEMENT_VOLUME, NO_BODY, NO_SLOT,
};
use crate::{SoftAttachmentRecord, SoftBodyRecord, SoftElementRecord, SoftParticleRecord};
use dynamis_model::{SoftElement, SoftElementKind, SoftElementState};

pub struct SoftParticleInit {
    pub position: [f32; 3],
    pub prev_position: [f32; 3],
    pub velocity: [f32; 3],
    pub radius: f32,
    pub inverse_mass: f32,
    pub friction: f32,
    pub support: f32,
    pub rest_spacing: f32,
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
            support: init.support,
            rest_spacing: init.rest_spacing,
            neighbour_offset: init.neighbour_offset,
            neighbour_count: init.neighbour_count,
            owner: init.owner,
            generation: init.generation,
            _wgsl_pad0: [0; 8],
        }
    }

    pub const fn cleared() -> Self {
        Self {
            position: [0.0; 4],
            prev_position: [0.0; 4],
            velocity: [0.0; 4],
            support: 0.0,
            rest_spacing: 0.0,
            neighbour_offset: NO_SLOT,
            neighbour_count: 0,
            owner: NO_BODY,
            generation: 0,
            _wgsl_pad0: [0; 8],
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

    pub const fn support(&self) -> f32 {
        self.support
    }

    pub const fn rest_spacing(&self) -> f32 {
        self.rest_spacing
    }

    pub const fn carries_continuum(&self) -> bool {
        self.support > 0.0
    }
}

impl SoftBodyRecord {
    pub const fn awake() -> Self {
        Self {
            sleep_timer: 0.0,
            sleeping: 0,
            moving: 0,
            wake: 0,
        }
    }

    pub const fn cleared() -> Self {
        Self {
            sleeping: 1,
            ..Self::awake()
        }
    }
}

pub struct SoftAttachmentInit {
    pub particle: u32,
    pub body_id: u32,
    pub generation: u32,
    pub local: [f32; 3],
}

impl SoftAttachmentRecord {
    pub fn build(init: SoftAttachmentInit) -> Self {
        Self {
            local: init.local,
            particle: init.particle,
            body_id: init.body_id,
            generation: init.generation,
            _wgsl_pad0: [0; 8],
        }
    }

    pub const fn cleared() -> Self {
        Self {
            local: [0.0; 3],
            particle: NO_SLOT,
            body_id: NO_BODY,
            generation: 0,
            _wgsl_pad0: [0; 8],
        }
    }
}

pub struct SoftElementInit {
    pub kind: u32,
    pub particles: [u32; ELEMENT_PARTICLES as usize],
    pub rest: f32,
    pub compliance: f32,
    pub yield_strain: f32,
    pub break_strain: f32,
    pub plastic_flow: f32,
}

impl SoftElementRecord {
    pub fn build(init: SoftElementInit) -> Self {
        Self {
            particles: init.particles,
            rest: init.rest,
            compliance: init.compliance,
            lambda: 0.0,
            kind: init.kind,
            yield_strain: init.yield_strain,
            break_strain: init.break_strain,
            plastic_flow: init.plastic_flow,
        }
    }

    pub const fn cleared() -> Self {
        Self {
            particles: [NO_SLOT; ELEMENT_PARTICLES as usize],
            rest: 0.0,
            compliance: 0.0,
            lambda: 0.0,
            kind: ELEMENT_DISTANCE,
            yield_strain: f32::INFINITY,
            break_strain: f32::INFINITY,
            plastic_flow: 0.0,
        }
    }

    pub fn state(&self) -> SoftElementState {
        SoftElementState::new(
            element_kind(self.kind),
            self.particles,
            self.rest,
            self.kind & ELEMENT_BROKEN != 0,
        )
    }
}

fn element_kind(kind: u32) -> SoftElementKind {
    match kind & ELEMENT_KIND_MASK {
        ELEMENT_DISTANCE => SoftElementKind::Distance,
        ELEMENT_AREA => SoftElementKind::Area,
        ELEMENT_BEND => SoftElementKind::Bend,
        ELEMENT_VOLUME => SoftElementKind::Volume,
        kind => panic!("soft element kind {kind} is outside the material table"),
    }
}

const _: () = assert!(ELEMENT_PARTICLES as usize == SoftElement::PARTICLES);
const _: () = assert!(SoftElement::UNUSED == NO_SLOT);
const _: () = assert!(SoftElementKind::Distance as u32 == ELEMENT_DISTANCE);
const _: () = assert!(SoftElementKind::Area as u32 == ELEMENT_AREA);
const _: () = assert!(SoftElementKind::Bend as u32 == ELEMENT_BEND);
const _: () = assert!(SoftElementKind::Volume as u32 == ELEMENT_VOLUME);
