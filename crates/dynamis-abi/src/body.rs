use crate::constant::{
    BODY_CCD, BODY_KINEMATIC, EDIT_ANGULAR_IMPULSE, EDIT_FORCE, EDIT_FORCE_AT_POINT, EDIT_IMPULSE,
    EDIT_IMPULSE_AT_POINT, EDIT_PATCH, EDIT_SLEEP, EDIT_TORQUE, EDIT_WAKE, OVERRIDE_SLEEP_ANGULAR,
    OVERRIDE_SLEEP_LINEAR,
};
use crate::{BodyDescriptorRecord, BodyEditRecord, BodyEditRunRecord, BodyStateRecord};
use bytemuck::Zeroable;
use dynamis_model::{BodyDesc, MassProperties, PhysicsConfig};

impl BodyStateRecord {
    pub fn initial(desc: &BodyDesc, body_id: u32, generation: u32) -> Self {
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
            force: [0.0; 3],
            _pad4: 0.0,
            torque: [0.0; 3],
            _pad5: 0.0,
            body_id,
            generation,
            sleep_timer: 0.0,
            sleeping: 0,
        }
    }

    pub fn observed(
        position: [f32; 3],
        prev_position: [f32; 3],
        orientation: [f32; 4],
        velocity: [f32; 3],
        angular_velocity: [f32; 3],
        body_id: u32,
        generation: u32,
    ) -> Self {
        Self {
            position,
            _pad0: 0.0,
            prev_position,
            _pad1: 0.0,
            orientation,
            velocity,
            _pad2: 0.0,
            angular_velocity,
            _pad3: 0.0,
            force: [0.0; 3],
            _pad4: 0.0,
            torque: [0.0; 3],
            _pad5: 0.0,
            body_id,
            generation,
            sleep_timer: 0.0,
            sleeping: 0,
        }
    }
}

impl BodyDescriptorRecord {
    pub fn build(desc: &BodyDesc, mass: MassProperties, config: &PhysicsConfig) -> Self {
        let mut flags = 0;
        if desc.kinematic {
            flags |= BODY_KINEMATIC;
        }
        if desc.ccd {
            flags |= BODY_CCD;
        }
        if desc.sleep_velocity.is_some() {
            flags |= OVERRIDE_SLEEP_LINEAR;
        }
        if desc.sleep_angular_velocity.is_some() {
            flags |= OVERRIDE_SLEEP_ANGULAR;
        }
        Self {
            inverse_mass: if desc.kinematic || desc.mass <= 0.0 {
                0.0
            } else {
                1.0 / desc.mass
            },
            linear_damping: desc.linear_damping.unwrap_or(config.damping),
            angular_damping: desc.angular_damping.unwrap_or(config.angular_damping),
            gravity_scale: desc.gravity_scale,
            sleep_velocity: desc.sleep_velocity.unwrap_or(0.0),
            sleep_angular_velocity: desc.sleep_angular_velocity.unwrap_or(0.0),
            flags,
            _pad0: 0,
            collision_group: desc.collision_group,
            collision_mask: desc.collision_mask,
            _pad1: 0,
            _pad4: 0,
            com: mass.com,
            _pad2: 0.0,
            inertia: mass.inertia,
            inverse_inertia: mass.inverse_inertia,
            _pad3: [0.0; 4],
        }
    }
}

impl BodyEditRunRecord {
    pub fn new(row: usize, first: usize, len: usize) -> Self {
        Self {
            row: row as u32,
            first: first as u32,
            len: len as u32,
            _pad: 0,
        }
    }
}

impl BodyEditRecord {
    fn edit(kind: u32, mask: u32, state: BodyStateRecord) -> Self {
        Self {
            kind,
            mask,
            _pad0: 0,
            _pad1: 0,
            state,
        }
    }

    pub fn patch(mask: u32, state: BodyStateRecord) -> Self {
        Self::edit(EDIT_PATCH, mask, state)
    }

    pub fn force(force: [f32; 3]) -> Self {
        let mut state = BodyStateRecord::zeroed();
        state.force = force;
        Self::edit(EDIT_FORCE, 0, state)
    }

    pub fn force_at_point(force: [f32; 3], point: [f32; 3]) -> Self {
        let mut state = BodyStateRecord::zeroed();
        state.force = force;
        state.position = point;
        Self::edit(EDIT_FORCE_AT_POINT, 0, state)
    }

    pub fn torque(torque: [f32; 3]) -> Self {
        let mut state = BodyStateRecord::zeroed();
        state.torque = torque;
        Self::edit(EDIT_TORQUE, 0, state)
    }

    pub fn impulse(impulse: [f32; 3]) -> Self {
        let mut state = BodyStateRecord::zeroed();
        state.velocity = impulse;
        Self::edit(EDIT_IMPULSE, 0, state)
    }

    pub fn impulse_at_point(impulse: [f32; 3], point: [f32; 3]) -> Self {
        let mut state = BodyStateRecord::zeroed();
        state.velocity = impulse;
        state.position = point;
        Self::edit(EDIT_IMPULSE_AT_POINT, 0, state)
    }

    pub fn angular_impulse(impulse: [f32; 3]) -> Self {
        let mut state = BodyStateRecord::zeroed();
        state.angular_velocity = impulse;
        Self::edit(EDIT_ANGULAR_IMPULSE, 0, state)
    }

    pub fn sleep() -> Self {
        Self::edit(EDIT_SLEEP, 0, BodyStateRecord::zeroed())
    }

    pub fn wake() -> Self {
        Self::edit(EDIT_WAKE, 0, BodyStateRecord::zeroed())
    }
}
