use crate::constant::{
    BODY_CCD, BODY_KINEMATIC, COMMAND_ADD, COMMAND_ANGULAR_IMPULSE, COMMAND_FORCE,
    COMMAND_FORCE_AT_POINT, COMMAND_IMPULSE, COMMAND_PATCH, COMMAND_REMOVE, COMMAND_SLEEP,
    COMMAND_SWAP, COMMAND_TORQUE, COMMAND_WAKE, IMPULSE_AT_POINT,
};
use bytemuck::{Pod, Zeroable};
use dynamis_model::{BodyDesc, MassProperties, PhysicsConfig};

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<RigidBodyRecord>() == 224);
    assert!(size_of::<BodyCommandRecord>() == 240);
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RigidBodyRecord {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub prev_position: [f32; 3],
    pub _pad1: f32,
    pub orientation: [f32; 4],
    pub velocity: [f32; 3],
    pub _pad2: f32,
    pub angular_velocity: [f32; 3],
    pub _pad3: f32,
    pub com: [f32; 3],
    pub _pad_com: f32,
    pub inverse_inertia_body: [f32; 6],
    pub _pad_inertia: [f32; 2],
    pub force: [f32; 3],
    pub _pad7: f32,
    pub torque: [f32; 3],
    pub _pad8: f32,
    pub inverse_mass: f32,
    pub restitution: f32,
    pub friction: f32,
    pub body_id: u32,
    pub generation: u32,
    pub collider_count: u32,
    pub flags: u32,
    pub collision_group: u32,
    pub collision_mask: u32,
    pub sleep_timer: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub sleep_velocity_override: f32,
    pub sleep_angular_velocity_override: f32,
    pub _pad_dynamics: f32,
}

impl RigidBodyRecord {
    pub fn build(
        desc: &BodyDesc,
        body_id: u32,
        generation: u32,
        mass: MassProperties,
        config: &PhysicsConfig,
    ) -> Self {
        let inverse_mass = desc.inverse_mass();
        Self {
            position: desc.position,
            _pad0: 0.0,
            prev_position: desc.position,
            _pad1: 0.0,
            orientation: desc.orientation,
            velocity: desc.velocity,
            _pad2: 0.0,
            angular_velocity: desc.angular_velocity,
            _pad3: 0.0,
            com: mass.com,
            _pad_com: 0.0,
            inverse_inertia_body: mass.inverse_inertia,
            _pad_inertia: [0.0; 2],
            force: [0.0; 3],
            _pad7: 0.0,
            torque: [0.0; 3],
            _pad8: 0.0,
            inverse_mass,
            restitution: desc.colliders[0].restitution,
            friction: desc.colliders[0].friction,
            body_id,
            generation,
            collider_count: desc.colliders.len() as u32,
            flags: matching_flags(desc),
            collision_group: desc.collision_group,
            collision_mask: desc.collision_mask,
            sleep_timer: 0.0,
            linear_damping: desc.linear_damping.unwrap_or(config.damping),
            angular_damping: desc.angular_damping.unwrap_or(config.angular_damping),
            gravity_scale: desc.gravity_scale,
            sleep_velocity_override: desc.sleep_velocity.unwrap_or(-1.0),
            sleep_angular_velocity_override: desc.sleep_angular_velocity.unwrap_or(-1.0),
            _pad_dynamics: 0.0,
        }
    }
}

pub fn matching_flags(desc: &BodyDesc) -> u32 {
    let mut flags = 0;
    if desc.kinematic {
        flags |= BODY_KINEMATIC;
    }
    if desc.ccd {
        flags |= BODY_CCD;
    }
    flags
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BodyCommandRecord {
    pub kind: u32,
    pub slot: u32,
    pub extra: u32,
    pub aux: u32,
    pub body: RigidBodyRecord,
}

impl BodyCommandRecord {
    pub fn add(slot: u32, body: RigidBodyRecord) -> Self {
        Self {
            kind: COMMAND_ADD,
            slot,
            extra: 0,
            aux: 0,
            body,
        }
    }

    pub fn remove(hole: u32, tail: u32) -> Self {
        Self {
            kind: COMMAND_REMOVE,
            slot: hole,
            extra: tail,
            aux: 0,
            body: RigidBodyRecord::zeroed(),
        }
    }

    pub fn swap(first: u32, second: u32) -> Self {
        Self {
            kind: COMMAND_SWAP,
            slot: first,
            extra: second,
            aux: 0,
            body: RigidBodyRecord::zeroed(),
        }
    }

    pub fn patch(slot: u32, mask: u32, body: RigidBodyRecord) -> Self {
        Self {
            kind: COMMAND_PATCH,
            slot,
            extra: mask,
            aux: 0,
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
            aux: 0,
            body,
        }
    }

    pub fn force_at_point(slot: u32, force: [f32; 3], point: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.force = force;
        body.position = point;
        Self {
            kind: COMMAND_FORCE_AT_POINT,
            slot,
            extra: 0,
            aux: 0,
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
            aux: 0,
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
            aux: 0,
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
            aux: 0,
            body,
        }
    }

    pub fn angular_impulse(slot: u32, impulse: [f32; 3]) -> Self {
        let mut body = RigidBodyRecord::zeroed();
        body.angular_velocity = impulse;
        Self {
            kind: COMMAND_ANGULAR_IMPULSE,
            slot,
            extra: 0,
            aux: 0,
            body,
        }
    }

    pub fn sleep(slot: u32) -> Self {
        Self {
            kind: COMMAND_SLEEP,
            slot,
            extra: 0,
            aux: 0,
            body: RigidBodyRecord::zeroed(),
        }
    }

    pub fn wake(slot: u32) -> Self {
        Self {
            kind: COMMAND_WAKE,
            slot,
            extra: 0,
            aux: 0,
            body: RigidBodyRecord::zeroed(),
        }
    }
}
