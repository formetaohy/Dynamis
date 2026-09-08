use crate::constant::{
    COMMAND_CONSTRAINT_ADD, COMMAND_CONSTRAINT_PATCH, COMMAND_CONSTRAINT_REMOVE, CONSTRAINT_BALL,
    CONSTRAINT_DISABLE_COLLISIONS, CONSTRAINT_DISTANCE, CONSTRAINT_FIXED, CONSTRAINT_GEAR,
    CONSTRAINT_HAS_BREAK, CONSTRAINT_HAS_LIMIT, CONSTRAINT_HAS_MOTOR, CONSTRAINT_HAS_SWING,
    CONSTRAINT_IS_SPRING, CONSTRAINT_PRISMATIC, CONSTRAINT_PULLEY, CONSTRAINT_REVOLUTE,
};
use bytemuck::{Pod, Zeroable};
use dynamis_model::ConstraintDesc;

const _: () = {
    use std::mem::size_of;
    assert!(size_of::<ConstraintRecord>() == 192);
    assert!(size_of::<ConstraintCommandRecord>() == 208);
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ConstraintRecord {
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
    pub accumulated: [f32; 8],
}

impl ConstraintRecord {
    pub fn build(desc: &ConstraintDesc, a: u32, b: u32) -> Self {
        let kind = match desc.kind {
            dynamis_model::ConstraintKind::Ball => CONSTRAINT_BALL,
            dynamis_model::ConstraintKind::Distance => CONSTRAINT_DISTANCE,
            dynamis_model::ConstraintKind::Revolute => CONSTRAINT_REVOLUTE,
            dynamis_model::ConstraintKind::Prismatic => CONSTRAINT_PRISMATIC,
            dynamis_model::ConstraintKind::Fixed => CONSTRAINT_FIXED,
            dynamis_model::ConstraintKind::Gear => CONSTRAINT_GEAR,
            dynamis_model::ConstraintKind::Pulley => CONSTRAINT_PULLEY,
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
            motor_speed: desc.motor.map_or(0.0, |motor| motor.target_velocity),
            motor_max_force: desc.motor.map_or(0.0, |motor| motor.max_force),
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
            accumulated: [0.0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ConstraintCommandRecord {
    pub kind: u32,
    pub slot: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub constraint: ConstraintRecord,
}

impl ConstraintCommandRecord {
    pub fn add(slot: u32, constraint: ConstraintRecord) -> Self {
        Self {
            kind: COMMAND_CONSTRAINT_ADD,
            slot,
            _pad0: 0,
            _pad1: 0,
            constraint,
        }
    }

    pub fn remove(slot: u32) -> Self {
        Self {
            kind: COMMAND_CONSTRAINT_REMOVE,
            slot,
            _pad0: 0,
            _pad1: 0,
            constraint: ConstraintRecord::zeroed(),
        }
    }

    pub fn patch(slot: u32, constraint: ConstraintRecord) -> Self {
        Self {
            kind: COMMAND_CONSTRAINT_PATCH,
            slot,
            _pad0: 0,
            _pad1: 0,
            constraint,
        }
    }
}
