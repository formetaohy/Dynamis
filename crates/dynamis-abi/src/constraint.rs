use crate::constant::{
    CONSTRAINT_ACCUMULATOR_SLOTS, CONSTRAINT_BALL, CONSTRAINT_CONE, CONSTRAINT_DISABLE_COLLISIONS,
    CONSTRAINT_DISTANCE, CONSTRAINT_FIXED, CONSTRAINT_GEAR, CONSTRAINT_HAS_BREAK,
    CONSTRAINT_HAS_LIMIT, CONSTRAINT_HAS_MOTOR, CONSTRAINT_HAS_SWING, CONSTRAINT_IS_SPRING,
    CONSTRAINT_PRISMATIC, CONSTRAINT_PULLEY, CONSTRAINT_REVOLUTE, CONSTRAINT_SIXDOF,
    CONSTRAINT_WARM_START, set_dof_driven, set_dof_limited, set_dof_locked,
};
use crate::{ConstraintDescriptorRecord, ConstraintRuntimeRecord};
use dynamis_model::{ConstraintData, ConstraintDesc, ConstraintKind, ConstraintMotor, DofDesc};

pub const fn kind_code(kind: ConstraintKind) -> u32 {
    match kind {
        ConstraintKind::Ball => CONSTRAINT_BALL,
        ConstraintKind::Distance => CONSTRAINT_DISTANCE,
        ConstraintKind::Revolute => CONSTRAINT_REVOLUTE,
        ConstraintKind::Prismatic => CONSTRAINT_PRISMATIC,
        ConstraintKind::Fixed => CONSTRAINT_FIXED,
        ConstraintKind::Gear => CONSTRAINT_GEAR,
        ConstraintKind::Pulley => CONSTRAINT_PULLEY,
        ConstraintKind::Cone => CONSTRAINT_CONE,
        ConstraintKind::SixDof => CONSTRAINT_SIXDOF,
    }
}

const _: () = {
    let mut seen = 0u32;
    let mut index = 0;
    while index < ConstraintKind::ALL.len() {
        let code = kind_code(ConstraintKind::ALL[index]);
        assert!(
            code <= CONSTRAINT_SIXDOF,
            "a joint kind must fit the declared device code space"
        );
        assert!(
            seen & (1 << code) == 0,
            "two joint kinds must not share a device code"
        );
        seen |= 1 << code;
        index += 1;
    }
    assert!(
        seen == (1 << (CONSTRAINT_SIXDOF + 1)) - 1,
        "every device joint code must belong to exactly one kind"
    );
};

pub(crate) fn emit_predicates(out: &mut String) {
    out.push_str("fn constraint_dof_count(kind: u32) -> u32 {\n");
    for kind in ConstraintKind::ALL {
        out.push_str(&format!(
            "    if (kind == {}u) {{ return {}u; }}\n",
            kind_code(*kind),
            kind.dofs().len(),
        ));
    }
    out.push_str("    return 0u;\n}\n");
}

fn pack_motor(record: &mut ConstraintDescriptorRecord, motor: ConstraintMotor) {
    record.motor_speed = motor.target_velocity();
    record.motor_max_force = motor.max_force();
    let position = motor.position_target();
    record.motor_position = position.map_or(0.0, |target| target.coordinate());
    record.motor_stiffness = position.map_or(0.0, |target| target.stiffness());
    record.motor_damping = position.map_or(0.0, |target| target.damping());
}

fn pack_dofs(record: &mut ConstraintDescriptorRecord, dofs: &[DofDesc; 6]) {
    for (index, dof) in dofs.iter().enumerate() {
        let flags = index as u32;
        record.flags = set_dof_locked(record.flags, flags, dof.locked);
        record.flags = set_dof_limited(record.flags, flags, dof.limit.is_some());
        record.flags = set_dof_driven(record.flags, flags, dof.motor.is_some());
        let lane = index % 3;
        if let Some(limit) = dof.limit {
            if index < 3 {
                record.linear_limit_min[lane] = limit.min;
                record.linear_limit_max[lane] = limit.max;
            } else {
                record.angular_limit_min[lane] = limit.min;
                record.angular_limit_max[lane] = limit.max;
            }
        }
        let Some(motor) = dof.motor else { continue };
        let position = motor.position_target();
        let speed = motor.target_velocity();
        let coordinate = position.map_or(0.0, |target| target.coordinate());
        let stiffness = position.map_or(0.0, |target| target.stiffness());
        let damping = position.map_or(0.0, |target| target.damping());
        let max_force = motor.max_force();
        if index < 3 {
            record.linear_motor_speed[lane] = speed;
            record.linear_motor_position[lane] = coordinate;
            record.linear_motor_stiffness[lane] = stiffness;
            record.linear_motor_damping[lane] = damping;
            record.linear_motor_force[lane] = max_force;
        } else {
            record.angular_motor_speed[lane] = speed;
            record.angular_motor_position[lane] = coordinate;
            record.angular_motor_stiffness[lane] = stiffness;
            record.angular_motor_damping[lane] = damping;
            record.angular_motor_force[lane] = max_force;
        }
    }
}

impl ConstraintDescriptorRecord {
    pub fn build(desc: &ConstraintDesc, first_body_id: u32, second_body_id: u32) -> Self {
        desc.assert_valid();
        let data = desc.data();
        let mut record = Self {
            kind: kind_code(data.kind()),
            first_body_id,
            second_body_id,
            swing_a: std::f32::consts::PI,
            swing_b: std::f32::consts::PI,
            gear_ratio: 1.0,
            ..bytemuck::Zeroable::zeroed()
        };
        if desc.disable_collisions_of() {
            record.flags |= CONSTRAINT_DISABLE_COLLISIONS;
        }
        if desc.warm_start_of() {
            record.flags |= CONSTRAINT_WARM_START;
        }
        if let Some(threshold) = desc.break_threshold_of() {
            record.flags |= CONSTRAINT_HAS_BREAK;
            record.break_force = threshold.force;
            record.break_torque = threshold.torque;
        }
        match data {
            ConstraintData::Ball {
                anchor_a,
                anchor_b,
                axis,
                twist,
                swing,
            } => {
                record.anchor_a = *anchor_a;
                record.anchor_b = *anchor_b;
                record.axis_a = *axis;
                if let Some(twist) = twist {
                    record.flags |= CONSTRAINT_HAS_LIMIT;
                    record.limit_min = twist.min;
                    record.limit_max = twist.max;
                }
                if let Some(swing) = swing {
                    record.flags |= CONSTRAINT_HAS_SWING;
                    record.swing_a = swing.swing_a;
                    record.swing_b = swing.swing_b;
                }
            }
            ConstraintData::Distance {
                anchor_a,
                anchor_b,
                length,
                spring,
            } => {
                record.anchor_a = *anchor_a;
                record.anchor_b = *anchor_b;
                record.distance = *length;
                if let Some(spring) = spring {
                    record.flags |= CONSTRAINT_IS_SPRING;
                    record.spring_frequency = spring.frequency;
                    record.spring_damping_ratio = spring.damping_ratio;
                }
            }
            ConstraintData::Revolute {
                anchor_a,
                anchor_b,
                axis,
                limit,
                motor,
            }
            | ConstraintData::Prismatic {
                anchor_a,
                anchor_b,
                axis,
                limit,
                motor,
            } => {
                record.anchor_a = *anchor_a;
                record.anchor_b = *anchor_b;
                record.axis_a = *axis;
                if let Some(limit) = limit {
                    record.flags |= CONSTRAINT_HAS_LIMIT;
                    record.limit_min = limit.min;
                    record.limit_max = limit.max;
                }
                if let Some(motor) = motor {
                    record.flags |= CONSTRAINT_HAS_MOTOR;
                    pack_motor(&mut record, *motor);
                }
            }
            ConstraintData::Fixed { anchor_a, anchor_b } => {
                record.anchor_a = *anchor_a;
                record.anchor_b = *anchor_b;
            }
            ConstraintData::Gear {
                axis_a,
                axis_b,
                ratio,
            } => {
                record.axis_a = *axis_a;
                record.axis_b = *axis_b;
                record.gear_ratio = *ratio;
            }
            ConstraintData::Pulley {
                anchor_a,
                anchor_b,
                fixed_a,
                fixed_b,
                length,
            } => {
                record.anchor_a = *anchor_a;
                record.anchor_b = *anchor_b;
                record.pulley_fixed_a = *fixed_a;
                record.pulley_fixed_b = *fixed_b;
                record.distance = *length;
            }
            ConstraintData::Cone {
                anchor_a,
                anchor_b,
                axis_a,
                axis_b,
                half_angle,
            } => {
                record.anchor_a = *anchor_a;
                record.anchor_b = *anchor_b;
                record.axis_a = *axis_a;
                record.axis_b = *axis_b;
                record.cone_angle = *half_angle;
            }
            ConstraintData::SixDof {
                anchor_a,
                anchor_b,
                axis,
                dofs,
            } => {
                record.anchor_a = *anchor_a;
                record.anchor_b = *anchor_b;
                record.axis_a = *axis;
                pack_dofs(&mut record, dofs);
            }
        }
        record
    }
}

impl ConstraintRuntimeRecord {
    pub fn fresh(constraint_id: u32, generation: u32) -> Self {
        Self {
            reaction: bytemuck::Zeroable::zeroed(),
            reference: [0.0; 4],
            accumulated: [0.0; CONSTRAINT_ACCUMULATOR_SLOTS as usize],
            broken: 0,
            constraint_id,
            generation,
            _pad0: 0,
        }
    }
}
