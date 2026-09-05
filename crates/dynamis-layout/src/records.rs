use bytemuck::{Pod, Zeroable};
use dynamis_model::BodyDesc;
use dynamis_model::PhysicsConfig;
use std::mem::size_of;

const _: () = {
    assert!(size_of::<RigidBodyRecord>() == 128);
    assert!(size_of::<SimParamsRecord>() == 48);
    assert!(size_of::<AabbRecord>() == 32);
    assert!(size_of::<PairRecord>() == 8);
    assert!(size_of::<ContactRecord>() == 32);
    assert!(size_of::<DispatchArgs>() == 12);
    assert!(size_of::<BodyCommandRecord>() == 144);
    assert!(size_of::<QueryRecord>() == 48);
    assert!(size_of::<QueryResultRecord>() == 16);
};

pub const COMMAND_ADD: u32 = 0;
pub const COMMAND_REMOVE: u32 = 1;
pub const COMMAND_PATCH: u32 = 2;
pub const COMMAND_FORCE: u32 = 3;
pub const COMMAND_TORQUE: u32 = 4;
pub const COMMAND_IMPULSE: u32 = 5;

pub const IMPULSE_AT_POINT: u32 = 1;

pub const PATCH_POSITION: u32 = 1;
pub const PATCH_VELOCITY: u32 = 2;
pub const PATCH_INVERSE_MASS: u32 = 4;
pub const PATCH_RADIUS: u32 = 8;
pub const PATCH_RESTITUTION: u32 = 16;
pub const PATCH_ORIENTATION: u32 = 32;
pub const PATCH_ANGULAR_VELOCITY: u32 = 64;
pub const PATCH_FRICTION: u32 = 128;

pub const QUERY_RAY: u32 = 0;
pub const QUERY_SPHERE: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RigidBodyRecord {
    pub position: [f32; 3],
    _pad0: f32,
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
    _pad1: f32,
    pub angular_velocity: [f32; 3],
    _pad2: f32,
    pub inverse_mass: f32,
    pub radius: f32,
    pub restitution: f32,
    pub friction: f32,
    pub body_id: u32,
    pub generation: u32,
    _pad5: f32,
    _pad6: f32,
    pub force: [f32; 3],
    _pad3: f32,
    pub torque: [f32; 3],
    _pad4: f32,
}

impl RigidBodyRecord {
    pub fn build(desc: &BodyDesc, body_id: u32, generation: u32) -> Self {
        Self {
            position: desc.position,
            _pad0: 0.0,
            orientation: desc.orientation,
            velocity: desc.velocity,
            _pad1: 0.0,
            angular_velocity: desc.angular_velocity,
            _pad2: 0.0,
            inverse_mass: if desc.mass > 0.0 {
                1.0 / desc.mass
            } else {
                0.0
            },
            radius: desc.radius,
            restitution: desc.restitution,
            friction: desc.friction,
            body_id,
            generation,
            _pad5: 0.0,
            _pad6: 0.0,
            force: [0.0; 3],
            _pad3: 0.0,
            torque: [0.0; 3],
            _pad4: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SimParamsRecord {
    pub gravity: [f32; 4],
    pub dt: f32,
    pub damping: f32,
    pub angular_damping: f32,
    pub body_count: u32,
    pub relaxation: f32,
    pub slop: f32,
    pub restitution_threshold: f32,
    _pad: f32,
}

impl SimParamsRecord {
    pub fn new(config: &PhysicsConfig, dt: f32, body_count: u32) -> Self {
        Self {
            gravity: [config.gravity[0], config.gravity[1], config.gravity[2], 0.0],
            dt,
            damping: config.damping,
            angular_damping: config.angular_damping,
            body_count,
            relaxation: config.relaxation,
            slop: config.slop,
            restitution_threshold: config.restitution_threshold,
            _pad: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AabbRecord {
    pub min: [f32; 3],
    _pad0: f32,
    pub max: [f32; 3],
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PairRecord {
    pub a: u32,
    pub b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ContactRecord {
    pub a: u32,
    pub b: u32,
    pub depth: f32,
    _pad: f32,
    pub normal: [f32; 3],
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DispatchArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl DispatchArgs {
    pub const fn none() -> Self {
        Self { x: 0, y: 1, z: 1 }
    }

    pub const fn sized(elements: u32) -> Self {
        Self {
            x: elements,
            y: 1,
            z: 1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BodyCommandRecord {
    pub kind: u32,
    pub slot: u32,
    pub extra: u32,
    _pad: u32,
    pub body: RigidBodyRecord,
}

impl BodyCommandRecord {
    pub fn add(slot: u32, body: RigidBodyRecord) -> Self {
        Self {
            kind: COMMAND_ADD,
            slot,
            extra: 0,
            _pad: 0,
            body,
        }
    }

    pub fn remove(hole: u32, tail: u32) -> Self {
        Self {
            kind: COMMAND_REMOVE,
            slot: hole,
            extra: tail,
            _pad: 0,
            body: RigidBodyRecord::zeroed(),
        }
    }

    pub fn patch(slot: u32, mask: u32, body: RigidBodyRecord) -> Self {
        Self {
            kind: COMMAND_PATCH,
            slot,
            extra: mask,
            _pad: 0,
            body,
        }
    }

    pub fn force(slot: u32, force: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.force = force;
        Self {
            kind: COMMAND_FORCE,
            slot,
            extra: 0,
            _pad: 0,
            body,
        }
    }

    pub fn torque(slot: u32, torque: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.torque = torque;
        Self {
            kind: COMMAND_TORQUE,
            slot,
            extra: 0,
            _pad: 0,
            body,
        }
    }

    pub fn impulse(slot: u32, impulse: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.velocity = impulse;
        Self {
            kind: COMMAND_IMPULSE,
            slot,
            extra: 0,
            _pad: 0,
            body,
        }
    }

    pub fn impulse_at_point(slot: u32, impulse: [f32; 3], point: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.velocity = impulse;
        body.position = point;
        Self {
            kind: COMMAND_IMPULSE,
            slot,
            extra: IMPULSE_AT_POINT,
            _pad: 0,
            body,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QueryRecord {
    pub origin: [f32; 3],
    pub kind: u32,
    pub direction: [f32; 3],
    pub extent: f32,
    pub slot: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl QueryRecord {
    pub fn ray(origin: [f32; 3], direction: [f32; 3], max_t: f32, slot: u32) -> Self {
        Self {
            origin,
            kind: QUERY_RAY,
            direction,
            extent: max_t,
            slot,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    pub fn sphere(center: [f32; 3], radius: f32, slot: u32) -> Self {
        Self {
            origin: center,
            kind: QUERY_SPHERE,
            direction: [0.0; 3],
            extent: radius,
            slot,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QueryResultRecord {
    pub body_id: u32,
    pub body_generation: u32,
    pub distance: f32,
    pub hit: u32,
}
