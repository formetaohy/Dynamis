fn joint_dof_count(kind: u32) -> u32 {
    if (kind == CONSTRAINT_FIXED || kind == CONSTRAINT_GEAR) {
        return 0u;
    }
    if (kind == CONSTRAINT_BALL) {
        return 3u;
    }
    if (kind == CONSTRAINT_SIXDOF) {
        return 6u;
    }
    return 1u;
}

fn joint_hinge_local(constraint: ConstraintDescriptor) -> vec3f {
    return normalize(constraint.axis_a);
}

fn joint_hinge(constraint: ConstraintDescriptor, first: Body) -> vec3f {
    return quat_rotate(first.state.orientation, joint_hinge_local(constraint));
}

fn joint_tangents_local(constraint: ConstraintDescriptor) -> TangentBasis {
    return make_tangents(joint_hinge_local(constraint));
}

fn joint_basis_local(constraint: ConstraintDescriptor, lane: u32) -> vec3f {
    return constraint_dof_axis(joint_tangents_local(constraint), joint_hinge_local(constraint), lane);
}

fn joint_dof_axis(constraint: ConstraintDescriptor, first: Body, dof: u32) -> vec3f {
    return sign_normalize(quat_rotate(first.state.orientation, joint_basis_local(constraint, dof % 3u)));
}

fn joint_axis(
    constraint: ConstraintDescriptor,
    first: Body,
    second: Body,
    anchor_a: vec3f,
    anchor_b: vec3f,
    dof: u32,
) -> vec3f {
    let kind = constraint.kind;
    if (kind == CONSTRAINT_DISTANCE) {
        return sign_normalize(anchor_b - anchor_a);
    }
    if (kind == CONSTRAINT_REVOLUTE || kind == CONSTRAINT_PRISMATIC) {
        return joint_hinge(constraint, first);
    }
    if (kind == CONSTRAINT_BALL) {
        if (dof < 2u) {
            return quat_rotate(first.state.orientation, joint_basis_local(constraint, dof));
        }
        return joint_hinge(constraint, first);
    }
    if (kind == CONSTRAINT_CONE) {
        let limb = normalize(quat_rotate(second.state.orientation, normalize(constraint.axis_b)));
        return sign_normalize(cross(joint_hinge(constraint, first), -limb));
    }
    if (kind == CONSTRAINT_PULLEY) {
        return sign_normalize(constraint.pulley_fixed_a - anchor_a);
    }
    if (kind == CONSTRAINT_SIXDOF) {
        return joint_dof_axis(constraint, first, dof);
    }
    return joint_hinge(constraint, first);
}

fn joint_coordinate(
    constraint: ConstraintDescriptor,
    first: Body,
    second: Body,
    anchor_a: vec3f,
    anchor_b: vec3f,
    dof: u32,
) -> f32 {
    let kind = constraint.kind;
    if (kind == CONSTRAINT_DISTANCE) {
        return length(anchor_b - anchor_a);
    }
    if (kind == CONSTRAINT_REVOLUTE) {
        return constraint_angle(first, second, joint_hinge_local(constraint));
    }
    if (kind == CONSTRAINT_PRISMATIC) {
        return dot(anchor_b - anchor_a, joint_hinge(constraint, first));
    }
    if (kind == CONSTRAINT_BALL) {
        let q_rel = quat_mul(quat_conjugate(first.state.orientation), second.state.orientation);
        if (dof < 2u) {
            let local_axis = joint_basis_local(constraint, dof);
            let perp = q_rel.xyz
                - joint_hinge_local(constraint) * dot(q_rel.xyz, joint_hinge_local(constraint));
            return 2.0 * atan2(dot(perp, local_axis), abs(q_rel.w));
        }
        return 2.0 * atan2(dot(q_rel.xyz, joint_hinge_local(constraint)), q_rel.w);
    }
    if (kind == CONSTRAINT_CONE) {
        let limb = normalize(quat_rotate(second.state.orientation, normalize(constraint.axis_b)));
        return acos(clamp(dot(joint_hinge(constraint, first), -limb), -1.0, 1.0));
    }
    if (kind == CONSTRAINT_PULLEY) {
        return length(anchor_a - constraint.pulley_fixed_a)
            + length(anchor_b - constraint.pulley_fixed_b);
    }
    if (kind == CONSTRAINT_SIXDOF) {
        let local_axis = joint_basis_local(constraint, dof % 3u);
        if (dof < 3u) {
            return dot(anchor_b - anchor_a, sign_normalize(quat_rotate(first.state.orientation, local_axis)));
        }
        return dot(constraint_relative_error(first, second, constraint.reference), local_axis);
    }
    return 0.0;
}

fn joint_angular_rate(first: Body, second: Body, axis: vec3f) -> f32 {
    return dot(second.state.angular_velocity - first.state.angular_velocity, normalize(axis));
}

fn joint_linear_rate(first: Body, second: Body, anchor_a: vec3f, anchor_b: vec3f, axis: vec3f) -> f32 {
    return dot(relative_velocity(first, second, anchor_a, anchor_b), normalize(axis));
}

fn joint_rate(
    constraint: ConstraintDescriptor,
    first: Body,
    second: Body,
    anchor_a: vec3f,
    anchor_b: vec3f,
    dof: u32,
) -> f32 {
    let kind = constraint.kind;
    if (kind == CONSTRAINT_DISTANCE || kind == CONSTRAINT_PRISMATIC) {
        return joint_linear_rate(
            first,
            second,
            anchor_a,
            anchor_b,
            joint_axis(constraint, first, second, anchor_a, anchor_b, dof),
        );
    }
    if (kind == CONSTRAINT_REVOLUTE) {
        return joint_angular_rate(first, second, joint_hinge(constraint, first));
    }
    if (kind == CONSTRAINT_BALL) {
        return joint_angular_rate(
            first,
            second,
            joint_axis(constraint, first, second, anchor_a, anchor_b, dof),
        );
    }
    if (kind == CONSTRAINT_SIXDOF) {
        let axis = joint_dof_axis(constraint, first, dof);
        if (dof < 3u) {
            return joint_linear_rate(first, second, anchor_a, anchor_b, axis);
        }
        return joint_angular_rate(first, second, axis);
    }
    if (kind == CONSTRAINT_CONE) {
        return joint_angular_rate(
            first,
            second,
            joint_axis(constraint, first, second, anchor_a, anchor_b, dof),
        );
    }
    if (kind == CONSTRAINT_PULLEY) {
        let dir_a = sign_normalize(constraint.pulley_fixed_a - anchor_a);
        let dir_b = sign_normalize(constraint.pulley_fixed_b - anchor_b);
        return dot(point_velocity(first, anchor_a), dir_a)
            + dot(point_velocity(second, anchor_b), dir_b);
    }
    return 0.0;
}

fn joint_impulse(constraint: ConstraintDescriptor, runtime: ConstraintRuntime, dof: u32) -> f32 {
    let kind = constraint.kind;
    if (kind == CONSTRAINT_DISTANCE || kind == CONSTRAINT_PULLEY) {
        return runtime.accumulated[0];
    }
    if (kind == CONSTRAINT_REVOLUTE || kind == CONSTRAINT_PRISMATIC) {
        return runtime.accumulated[6] + runtime.accumulated[5];
    }
    if (kind == CONSTRAINT_BALL) {
        if (dof < 2u) {
            return runtime.accumulated[4u + dof];
        }
        return 0.0;
    }
    if (kind == CONSTRAINT_CONE) {
        return runtime.accumulated[3];
    }
    if (kind == CONSTRAINT_SIXDOF) {
        return runtime.accumulated[dof] + runtime.accumulated[DOF_LIMIT_ROW_BASE + dof];
    }
    return 0.0;
}
