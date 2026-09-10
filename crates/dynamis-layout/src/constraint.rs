use crate::constant::{
    COMMAND_CONSTRAINT_ADD, COMMAND_CONSTRAINT_SWAP, CONSTRAINT_BALL, CONSTRAINT_CONE,
    CONSTRAINT_DISABLE_COLLISIONS, CONSTRAINT_DISTANCE, CONSTRAINT_FIXED, CONSTRAINT_GEAR,
    CONSTRAINT_HAS_BREAK, CONSTRAINT_HAS_LIMIT, CONSTRAINT_HAS_MOTOR, CONSTRAINT_HAS_SWING,
    CONSTRAINT_IS_SPRING, CONSTRAINT_PRISMATIC, CONSTRAINT_PULLEY, CONSTRAINT_REVOLUTE,
    CONSTRAINT_SIXDOF, CONSTRAINT_WARM_START, DOF_DRIVEN, DOF_FREE, DOF_LIMITED, DOF_LOCKED,
    set_dof_mode,
};
use bytemuck::{Pod, Zeroable};
use dynamis_model::{ConstraintDesc, ConstraintMotor, DofDesc};

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<ConstraintDescriptorRecord>() == 384);
    assert!(size_of::<ConstraintRuntimeRecord>() == 48);
    assert!(size_of::<ConstraintCommandRecord>() == 24);
};

/// Joint parameters of one constraint slot. The host owns this row and the
/// device reads it; it never appears as a write target in a shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ConstraintDescriptorRecord {
    pub kind: u32,
    pub a: u32,
    pub b: u32,
    pub flags: u32,
    pub anchor_a: [f32; 3],
    pub _pad1: f32,
    pub anchor_b: [f32; 3],
    pub _pad2: f32,
    pub axis_a: [f32; 3],
    pub _pad3: f32,
    pub axis_b: [f32; 3],
    pub _pad4: f32,
    pub distance: f32,
    pub limit_min: f32,
    pub limit_max: f32,
    pub swing_a: f32,
    pub swing_b: f32,
    pub motor_speed: f32,
    pub motor_max_force: f32,
    pub spring_frequency: f32,
    pub spring_damping_ratio: f32,
    pub break_force: f32,
    pub break_torque: f32,
    pub gear_ratio: f32,
    pub pulley_fixed_a: [f32; 3],
    pub _pad_pulley_a: f32,
    pub pulley_fixed_b: [f32; 3],
    pub _pad_pulley_b: f32,
    pub motor_target: f32,
    pub motor_stiffness: f32,
    pub motor_damping: f32,
    pub cone_angle: f32,
    pub reference: [f32; 4],
    pub linear_limit_min: [f32; 3],
    pub _pad_lim_min: f32,
    pub linear_limit_max: [f32; 3],
    pub _pad_lim_max: f32,
    pub angular_limit_min: [f32; 3],
    pub _pad_ang_min: f32,
    pub angular_limit_max: [f32; 3],
    pub _pad_ang_max: f32,
    pub linear_motor_target: [f32; 3],
    pub _pad_lin_target: f32,
    pub linear_motor_stiffness: [f32; 3],
    pub _pad_lin_stiff: f32,
    pub linear_motor_damping: [f32; 3],
    pub _pad_lin_damp: f32,
    pub angular_motor_target: [f32; 3],
    pub _pad_ang_target: f32,
    pub angular_motor_stiffness: [f32; 3],
    pub _pad_ang_stiff: f32,
    pub angular_motor_damping: [f32; 3],
    pub _pad_ang_damp: f32,
    pub linear_motor_force: [f32; 3],
    pub _pad_lin_force: f32,
    pub angular_motor_force: [f32; 3],
    pub _pad_ang_force: f32,
}

impl ConstraintDescriptorRecord {
    pub fn build(desc: &ConstraintDesc, a: u32, b: u32) -> Self {
        let kind = match desc.kind {
            dynamis_model::ConstraintKind::Ball => CONSTRAINT_BALL,
            dynamis_model::ConstraintKind::Distance => CONSTRAINT_DISTANCE,
            dynamis_model::ConstraintKind::Revolute => CONSTRAINT_REVOLUTE,
            dynamis_model::ConstraintKind::Prismatic => CONSTRAINT_PRISMATIC,
            dynamis_model::ConstraintKind::Fixed => CONSTRAINT_FIXED,
            dynamis_model::ConstraintKind::Gear => CONSTRAINT_GEAR,
            dynamis_model::ConstraintKind::Pulley => CONSTRAINT_PULLEY,
            dynamis_model::ConstraintKind::Cone => CONSTRAINT_CONE,
            dynamis_model::ConstraintKind::SixDof => CONSTRAINT_SIXDOF,
        };
        let mut flags = 0;
        if desc.disable_collisions {
            flags |= CONSTRAINT_DISABLE_COLLISIONS;
        }
        if desc.limit.is_some() {
            flags |= CONSTRAINT_HAS_LIMIT;
        }
        if desc.swing.is_some() {
            flags |= CONSTRAINT_HAS_SWING;
        }
        if desc.motor.is_some() {
            flags |= CONSTRAINT_HAS_MOTOR;
        }
        if desc.spring.is_some() {
            flags |= CONSTRAINT_IS_SPRING;
        }
        if desc.break_threshold.is_some() {
            flags |= CONSTRAINT_HAS_BREAK;
        }
        if desc.warm_start {
            flags |= CONSTRAINT_WARM_START;
        }
        let dofs = desc.dofs.unwrap_or([DofDesc::free(); 6]);
        for (index, dof) in dofs.iter().enumerate() {
            let mode = if dof.locked {
                DOF_LOCKED
            } else if dof.limit.is_some() {
                DOF_LIMITED
            } else if dof.motor.is_some() {
                DOF_DRIVEN
            } else {
                DOF_FREE
            };
            flags = set_dof_mode(flags, index as u32, mode);
        }
        let motor = |motor: &Option<ConstraintMotor>| {
            (
                motor.map_or(0.0, |m| m.target_velocity),
                motor.map_or(0.0, |m| m.max_force),
                motor.map_or(0.0, |m| m.target_position.unwrap_or(0.0)),
                motor.map_or(0.0, |m| m.stiffness),
                motor.map_or(0.0, |m| m.damping),
            )
        };
        let (motor_speed, motor_max_force, motor_target, motor_stiffness, motor_damping) =
            motor(&desc.motor);
        let generic_motor = |dofs: &[DofDesc; 6]| {
            let mut target = [0.0f32; 3];
            let mut stiffness = [0.0f32; 3];
            let mut damping = [0.0f32; 3];
            let mut force = [0.0f32; 3];
            let mut angular_target = [0.0f32; 3];
            let mut angular_stiffness = [0.0f32; 3];
            let mut angular_damping = [0.0f32; 3];
            let mut angular_force = [0.0f32; 3];
            for (index, dof) in dofs.iter().enumerate() {
                let Some(motor) = dof.motor else { continue };
                if index < 3 {
                    target[index] = motor.target_position.unwrap_or(motor.target_velocity);
                    stiffness[index] = motor.stiffness;
                    damping[index] = motor.damping;
                    force[index] = motor.max_force;
                } else {
                    angular_target[index - 3] =
                        motor.target_position.unwrap_or(motor.target_velocity);
                    angular_stiffness[index - 3] = motor.stiffness;
                    angular_damping[index - 3] = motor.damping;
                    angular_force[index - 3] = motor.max_force;
                }
            }
            (
                target,
                stiffness,
                damping,
                force,
                angular_target,
                angular_stiffness,
                angular_damping,
                angular_force,
            )
        };
        let (
            linear_motor_target,
            linear_motor_stiffness,
            linear_motor_damping,
            linear_motor_force,
            angular_motor_target,
            angular_motor_stiffness,
            angular_motor_damping,
            angular_motor_force,
        ) = generic_motor(&dofs);
        let mut linear_limits: [[f32; 3]; 2] = [[0.0; 3]; 2];
        let mut angular_limits: [[f32; 3]; 2] = [[0.0; 3]; 2];
        for (index, dof) in dofs.iter().enumerate() {
            let Some(limit) = dof.limit else { continue };
            if index < 3 {
                linear_limits[0][index] = limit.min;
                linear_limits[1][index] = limit.max;
            } else {
                angular_limits[0][index - 3] = limit.min;
                angular_limits[1][index - 3] = limit.max;
            }
        }
        Self {
            kind,
            a,
            b,
            flags,
            anchor_a: desc.anchor_a,
            _pad1: 0.0,
            anchor_b: desc.anchor_b,
            _pad2: 0.0,
            axis_a: desc.axis_a,
            _pad3: 0.0,
            axis_b: desc.axis_b,
            _pad4: 0.0,
            distance: desc.rest_length,
            limit_min: desc.limit.map_or(0.0, |limit| limit.min),
            limit_max: desc.limit.map_or(0.0, |limit| limit.max),
            swing_a: desc
                .swing
                .map_or(std::f32::consts::PI, |swing| swing.swing_a),
            swing_b: desc
                .swing
                .map_or(std::f32::consts::PI, |swing| swing.swing_b),
            motor_speed,
            motor_max_force,
            spring_frequency: desc.spring.map_or(0.0, |spring| spring.frequency),
            spring_damping_ratio: desc.spring.map_or(0.0, |spring| spring.damping_ratio),
            break_force: desc
                .break_threshold
                .map_or(0.0, |threshold| threshold.force),
            break_torque: desc
                .break_threshold
                .map_or(0.0, |threshold| threshold.torque),
            gear_ratio: desc.gear_ratio,
            pulley_fixed_a: desc.pulley_fixed_a,
            _pad_pulley_a: 0.0,
            pulley_fixed_b: desc.pulley_fixed_b,
            _pad_pulley_b: 0.0,
            motor_target,
            motor_stiffness,
            motor_damping,
            cone_angle: desc.cone_angle,
            reference: desc.reference,
            linear_limit_min: linear_limits[0],
            _pad_lim_min: 0.0,
            linear_limit_max: linear_limits[1],
            _pad_lim_max: 0.0,
            angular_limit_min: angular_limits[0],
            _pad_ang_min: 0.0,
            angular_limit_max: angular_limits[1],
            _pad_ang_max: 0.0,
            linear_motor_target,
            _pad_lin_target: 0.0,
            linear_motor_stiffness,
            _pad_lin_stiff: 0.0,
            linear_motor_damping,
            _pad_lin_damp: 0.0,
            angular_motor_target,
            _pad_ang_target: 0.0,
            angular_motor_stiffness,
            _pad_ang_stiff: 0.0,
            angular_motor_damping,
            _pad_ang_damp: 0.0,
            linear_motor_force,
            _pad_lin_force: 0.0,
            angular_motor_force,
            _pad_ang_force: 0.0,
        }
    }
}

/// Accumulated impulses of one constraint slot. The device owns this row; the
/// host only reads it back to report breaks.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ConstraintRuntimeRecord {
    pub accumulated: [f32; 8],
    pub broken: u32,
    pub constraint_id: u32,
    pub generation: u32,
    pub _pad0: u32,
}

impl ConstraintRuntimeRecord {
    /// The runtime row a fresh constraint slot starts with.
    pub fn ignited(constraint_id: u32, generation: u32) -> Self {
        Self {
            accumulated: [0.0; 8],
            broken: 0,
            constraint_id,
            generation,
            _pad0: 0,
        }
    }
}

impl ConstraintRuntimeRecord {
    pub fn fresh(id: u32, generation: u32) -> Self {
        Self {
            accumulated: [0.0; 8],
            broken: 0,
            constraint_id: id,
            generation,
            _pad0: 0,
        }
    }
}

/// A pending mutation of [`ConstraintRuntimeRecord`] at a slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ConstraintCommandRecord {
    pub kind: u32,
    pub slot: u32,
    pub tail: u32,
    pub constraint_id: u32,
    pub generation: u32,
    pub _pad0: u32,
}

impl ConstraintCommandRecord {
    pub fn add(slot: u32, id: u32, generation: u32) -> Self {
        Self {
            kind: COMMAND_CONSTRAINT_ADD,
            slot,
            tail: 0,
            constraint_id: id,
            generation,
            _pad0: 0,
        }
    }

    pub fn swap(slot: u32, tail: u32) -> Self {
        Self {
            kind: COMMAND_CONSTRAINT_SWAP,
            slot,
            tail,
            constraint_id: 0,
            generation: 0,
            _pad0: 0,
        }
    }
}
