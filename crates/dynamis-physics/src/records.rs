use crate::body::BodyDesc;
use bytemuck::{Pod, Zeroable};
use std::mem::size_of;

const _: () = {
    assert!(size_of::<RigidBodyRecord>() == 48);
    assert!(size_of::<SimParamsRecord>() == 32);
    assert!(size_of::<AabbRecord>() == 32);
    assert!(size_of::<PairRecord>() == 8);
    assert!(size_of::<ContactRecord>() == 32);
    assert!(size_of::<DispatchCount>() == 12);
    assert!(size_of::<BodyCommandRecord>() == 64);
};

pub(crate) const COMMAND_ADD: u32 = 0;
pub(crate) const COMMAND_REMOVE: u32 = 1;
pub(crate) const COMMAND_PATCH: u32 = 2;

pub(crate) const PATCH_POSITION: u32 = 1;
pub(crate) const PATCH_VELOCITY: u32 = 2;
pub(crate) const PATCH_INVERSE_MASS: u32 = 4;
pub(crate) const PATCH_RADIUS: u32 = 8;
pub(crate) const PATCH_RESTITUTION: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct RigidBodyRecord {
    pub(crate) position: [f32; 3],
    _pad0: f32,
    pub(crate) velocity: [f32; 3],
    _pad1: f32,
    pub(crate) inverse_mass: f32,
    pub(crate) radius: f32,
    pub(crate) restitution: f32,
    _pad2: f32,
}

impl RigidBodyRecord {
    pub(crate) fn build(desc: &BodyDesc) -> Self {
        Self {
            position: desc.position,
            _pad0: 0.0,
            velocity: desc.velocity,
            _pad1: 0.0,
            inverse_mass: if desc.mass > 0.0 { 1.0 / desc.mass } else { 0.0 },
            radius: desc.radius,
            restitution: desc.restitution,
            _pad2: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SimParamsRecord {
    pub(crate) gravity: [f32; 4],
    pub(crate) dt: f32,
    pub(crate) damping: f32,
    pub(crate) body_count: u32,
    pub(crate) relaxation: f32,
}

impl SimParamsRecord {
    pub(crate) fn new(
        gravity: [f32; 3],
        dt: f32,
        damping: f32,
        body_count: u32,
        relaxation: f32,
    ) -> Self {
        Self {
            gravity: [gravity[0], gravity[1], gravity[2], 0.0],
            dt,
            damping,
            body_count,
            relaxation,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct AabbRecord {
    pub(crate) min: [f32; 3],
    _pad0: f32,
    pub(crate) max: [f32; 3],
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct PairRecord {
    pub(crate) a: u32,
    pub(crate) b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ContactRecord {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) depth: f32,
    _pad: f32,
    pub(crate) normal: [f32; 3],
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct DispatchCount {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) z: u32,
}

impl DispatchCount {
    pub(crate) const fn idle() -> Self {
        Self { x: 0, y: 1, z: 1 }
    }

    pub(crate) const fn sized(elements: u32) -> Self {
        Self {
            x: elements,
            y: 1,
            z: 1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct BodyCommandRecord {
    pub(crate) kind: u32,
    pub(crate) slot: u32,
    pub(crate) extra: u32,
    _pad: u32,
    pub(crate) record: RigidBodyRecord,
}

impl BodyCommandRecord {
    pub(crate) fn add(slot: u32, record: RigidBodyRecord) -> Self {
        Self {
            kind: COMMAND_ADD,
            slot,
            extra: 0,
            _pad: 0,
            record,
        }
    }

    pub(crate) fn remove(hole: u32, tail: u32) -> Self {
        Self {
            kind: COMMAND_REMOVE,
            slot: hole,
            extra: tail,
            _pad: 0,
            record: RigidBodyRecord::zeroed(),
        }
    }

    pub(crate) fn patch(slot: u32, mask: u32, record: RigidBodyRecord) -> Self {
        Self {
            kind: COMMAND_PATCH,
            slot,
            extra: mask,
            _pad: 0,
            record,
        }
    }
}
