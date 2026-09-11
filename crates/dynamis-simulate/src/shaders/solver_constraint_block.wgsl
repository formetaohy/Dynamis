struct FrameForces {
    linear: f32,
    angular: f32,
}

struct RowOutcome {
    first: Body,
    second: Body,
    accumulated: f32,
    forces: FrameForces,
}

fn constraint_anchor(body: Body, local: vec3f) -> vec3f {
    return body.state.position + quat_rotate(body.state.orientation, local);
}

fn constraint_breach(anchor_a: vec3f, anchor_b: vec3f, constraint: ConstraintDescriptor) -> bool {
    if (constraint.kind == CONSTRAINT_GEAR) {
        return false;
    }
    if (constraint.kind == CONSTRAINT_PULLEY) {
        let length = length(anchor_a - constraint.pulley_fixed_a) + length(anchor_b - constraint.pulley_fixed_b);
        return abs(length - constraint.distance) > 0.05;
    }
    let separation = length(anchor_b - anchor_a);
    return abs(separation - constraint.distance) > 0.05;
}

fn solve_point_row(
    axis: vec3f,
    point_a: vec3f,
    point_b: vec3f,
    goal: f32,
    max_bias: f32,
    mass_first: Body,
    mass_second: Body,
    first: Body,
    second: Body,
    accumulated: f32,
    forces: FrameForces,
) -> RowOutcome {
    let bias = dot(point_b - point_a, axis) - goal;
    let k = point_momentum_mass(mass_first, mass_second, point_a, point_b, axis);
    var first_out = first;
    var second_out = second;
    var accumulated_out = accumulated;
    var forces_out = forces;
    if (k > 0.0) {
        let velocity = relative_velocity(first, second, point_a, point_b);
        let jacobian_speed = dot(velocity, axis);
        let bias_speed = clamp(bias * 0.05 / max(params.dt, 1e-4), -max_bias, max_bias);
        let next = accumulated - (jacobian_speed + bias_speed) / k;
        let applied = next - accumulated;
        apply_pair_impulse(&first_out, &second_out, point_a, point_b, axis * applied);
        accumulated_out = next;
        forces_out.linear = forces_out.linear + abs(applied);
    }
    var outcome: RowOutcome;
    outcome.first = first_out;
    outcome.second = second_out;
    outcome.accumulated = accumulated_out;
    outcome.forces = forces_out;
    return outcome;
}

fn solve_angular_row(
    axis: vec3f,
    mass_first: Body,
    mass_second: Body,
    first: Body,
    second: Body,
    accumulated: f32,
    bias_speed: f32,
    forces: FrameForces,
) -> RowOutcome {
    let axis_normalized = normalize(axis);
    let k = dot(
        axis_normalized,
        apply_inverse_inertia(mass_first, axis_normalized) + apply_inverse_inertia(mass_second, axis_normalized),
    );
    var first_out = first;
    var second_out = second;
    var accumulated_out = accumulated;
    var forces_out = forces;
    if (k > 0.0) {
        let velocity = second.state.angular_velocity - first.state.angular_velocity;
        let jacobian_speed = dot(velocity, axis_normalized);
        let next = accumulated - (jacobian_speed + bias_speed) / k;
        let applied = next - accumulated;
        first_out.state.angular_velocity =
            first.state.angular_velocity - apply_inverse_inertia(first, axis_normalized * applied);
        second_out.state.angular_velocity =
            second.state.angular_velocity + apply_inverse_inertia(second, axis_normalized * applied);
        accumulated_out = next;
        forces_out.angular = forces_out.angular + abs(applied);
    }
    var outcome: RowOutcome;
    outcome.first = first_out;
    outcome.second = second_out;
    outcome.accumulated = accumulated_out;
    outcome.forces = forces_out;
    return outcome;
}

fn solve_driven_point_row(
    axis: vec3f,
    point_a: vec3f,
    point_b: vec3f,
    current: f32,
    target_value: f32,
    stiffness: f32,
    damping: f32,
    target_speed: f32,
    max_force: f32,
    mass_first: Body,
    mass_second: Body,
    first: Body,
    second: Body,
    accumulated: f32,
    forces: FrameForces,
) -> RowOutcome {
    let k = point_momentum_mass(mass_first, mass_second, point_a, point_b, axis);
    var first_out = first;
    var second_out = second;
    var accumulated_out = accumulated;
    var forces_out = forces;
    if (k > 0.0) {
        let velocity = relative_velocity(first, second, point_a, point_b);
        let jacobian_speed = dot(velocity, axis);
        let servo_speed =
            stiffness * (target_value - current) / max(params.dt, 1e-4) - damping * jacobian_speed;
        let v_target = target_speed + servo_speed;
        let next = accumulated - (jacobian_speed - v_target) / k;
        let cap = max_force * params.dt;
        let applied_raw = next - accumulated;
        let applied = select(applied_raw, clamp(applied_raw, -cap, cap), cap > 0.0);
        apply_pair_impulse(&first_out, &second_out, point_a, point_b, axis * applied);
        accumulated_out = accumulated + applied;
        forces_out.linear = forces_out.linear + abs(applied);
    }
    var outcome: RowOutcome;
    outcome.first = first_out;
    outcome.second = second_out;
    outcome.accumulated = accumulated_out;
    outcome.forces = forces_out;
    return outcome;
}

fn solve_driven_angular_row(
    axis: vec3f,
    current: f32,
    target_value: f32,
    stiffness: f32,
    damping: f32,
    target_speed: f32,
    max_force: f32,
    mass_first: Body,
    mass_second: Body,
    first: Body,
    second: Body,
    accumulated: f32,
    forces: FrameForces,
) -> RowOutcome {
    let axis_normalized = normalize(axis);
    let k = dot(
        axis_normalized,
        apply_inverse_inertia(mass_first, axis_normalized) + apply_inverse_inertia(mass_second, axis_normalized),
    );
    var first_out = first;
    var second_out = second;
    var accumulated_out = accumulated;
    var forces_out = forces;
    if (k > 0.0) {
        let velocity = second.state.angular_velocity - first.state.angular_velocity;
        let jacobian_speed = dot(velocity, axis_normalized);
        let servo_speed =
            stiffness * (target_value - current) / max(params.dt, 1e-4) - damping * jacobian_speed;
        let v_target = target_speed + servo_speed;
        let next = accumulated - (jacobian_speed - v_target) / k;
        let cap = max_force * params.dt;
        let applied_raw = next - accumulated;
        let applied = select(applied_raw, clamp(applied_raw, -cap, cap), cap > 0.0);
        first_out.state.angular_velocity =
            first.state.angular_velocity - apply_inverse_inertia(first, axis_normalized * applied);
        second_out.state.angular_velocity =
            second.state.angular_velocity + apply_inverse_inertia(second, axis_normalized * applied);
        accumulated_out = accumulated + applied;
        forces_out.angular = forces_out.angular + abs(applied);
    }
    var outcome: RowOutcome;
    outcome.first = first_out;
    outcome.second = second_out;
    outcome.accumulated = accumulated_out;
    outcome.forces = forces_out;
    return outcome;
}

fn hinge_axis(constraint: ConstraintDescriptor, first: Body) -> vec3f {
    return normalize(quat_rotate(first.state.orientation, constraint.axis_a));
}

fn relative_angle(first: Body, second: Body, hinge: vec3f) -> f32 {
    let q_rel = quat_mul(quat_conjugate(first.state.orientation), second.state.orientation);
    let signed = atan2(dot(q_rel.xyz, hinge), q_rel.w) * 2.0;
    return signed;
}

fn limit_row(
    error: f32,
    accumulated: f32,
    axis: vec3f,
    mass_first: Body,
    mass_second: Body,
    first: Body,
    second: Body,
    forces: FrameForces,
) -> RowOutcome {
    let bias_speed = clamp(error * 0.05 / max(params.dt, 1e-4), -4.0, 4.0);
    let outcome = solve_angular_row(axis, mass_first, mass_second, first, second, accumulated, bias_speed, forces);
    return outcome;
}

fn dof_mode(flags: u32, index: u32) -> u32 {
    return (flags >> (8u + index * 2u)) & 3u;
}

fn vec_index(v: vec3f, index: u32) -> f32 {
    return select(select(v.x, v.z, index == 2u), v.y, index == 1u);
}

fn dof_frame(tangents: TangentBasis, axis_a: vec3f, index: u32) -> vec3f {
    return select(tangents.first, select(tangents.second, axis_a, index == 2u), index == 1u);
}

fn solve_constraint_block(constraint_index: u32, slot: u32) {
    var runtime = constraint_runtime[constraint_index];
    let constraint = constraint_descs[constraint_index];
    if (runtime.broken != 0u) {
        store_block_delta(slot, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
        return;
    }
    let pair = block_pair(constraint.a, constraint.b);
    if (body_is_inert(pair.first) && body_is_inert(pair.second)) {
        store_block_delta(slot, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
        return;
    }
    var first = pair.first;
    var second = pair.second;
    let mass_first = pair.split_first;
    let mass_second = pair.split_second;
    let first_sleeping = pair.first.state.sleeping != 0u;
    let second_sleeping = pair.second.state.sleeping != 0u;
    let anchor_a = constraint_anchor(first, constraint.anchor_a);
    let anchor_b = constraint_anchor(second, constraint.anchor_b);
    var accumulated = runtime.accumulated;
    var forces: FrameForces;
    forces.linear = 0.0;
    forces.angular = 0.0;
    let max_bias = 4.0;
    if (constraint.kind == CONSTRAINT_DISTANCE) {
        let axis = sign_normalize(anchor_b - anchor_a);
        if ((constraint.flags & CONSTRAINT_IS_SPRING) == 0u) {
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, constraint.distance, max_bias,
                mass_first, mass_second, first, second, accumulated[0], forces,
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[0] = outcome.accumulated;
            forces = outcome.forces;
        } else {
            let separation = length(anchor_b - anchor_a);
            let error = separation - constraint.distance;
            let omega = 6.2831853 * constraint.spring_frequency;
            let zeta = constraint.spring_damping_ratio;
            let velocity = relative_velocity(first, second, anchor_a, anchor_b);
            let jacobian_speed = dot(velocity, axis);
            let k = point_momentum_mass(mass_first, mass_second, anchor_a, anchor_b, axis);
            if (k > 0.0) {
                let v_target = -(omega * omega * error * params.dt) - 2.0 * zeta * omega * params.dt * jacobian_speed;
                let next = accumulated[0] - (jacobian_speed - v_target) / k;
                let applied = next - accumulated[0];
                apply_pair_impulse(&first, &second, anchor_a, anchor_b, axis * applied);
                accumulated[0] = next;
                forces.linear = forces.linear + abs(applied);
            }
        }
    } else if (constraint.kind == CONSTRAINT_REVOLUTE) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = select(vec3f(1.0, 0.0, 0.0), select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), i == 2u), i == 1u);
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, 0.0, max_bias,
                mass_first, mass_second, first, second, accumulated[i], forces,
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
            forces = outcome.forces;
        }
        let hinge = hinge_axis(constraint, first);
        let tangents = make_tangents(hinge);
        for (var i = 0u; i < 2u; i = i + 1u) {
            let axis = select(tangents.first, tangents.second, i == 1u);
            let outcome = solve_angular_row(axis, mass_first, mass_second, first, second, accumulated[3u + i], 0.0, forces);
            first = outcome.first;
            second = outcome.second;
            accumulated[3u + i] = outcome.accumulated;
            forces = outcome.forces;
        }
        if ((constraint.flags & CONSTRAINT_HAS_MOTOR) != 0u) {
            let current = relative_angle(first, second, hinge);
            let outcome = solve_driven_angular_row(
                hinge, current, constraint.motor_target, constraint.motor_stiffness, constraint.motor_damping,
                -constraint.motor_speed, constraint.motor_max_force,
                mass_first, mass_second, first, second, accumulated[6], forces,
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[6] = outcome.accumulated;
            forces = outcome.forces;
        }
        if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
            let angle = relative_angle(first, second, hinge);
            let above = angle > constraint.limit_max;
            let below = angle < constraint.limit_min;
            if (above) {
                let error = angle - constraint.limit_max;
                let outcome = limit_row(error, accumulated[5], hinge, mass_first, mass_second, first, second, forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[5] = min(outcome.accumulated, 0.0);
                forces = outcome.forces;
            } else if (below) {
                let error = angle - constraint.limit_min;
                let outcome = limit_row(error, accumulated[5], hinge, mass_first, mass_second, first, second, forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[5] = max(outcome.accumulated, 0.0);
                forces = outcome.forces;
            }
        }
    } else if (constraint.kind == CONSTRAINT_PRISMATIC) {
        let axis = normalize(quat_rotate(first.state.orientation, constraint.axis_a));
        let tangents = make_tangents(axis);
        for (var i = 0u; i < 2u; i = i + 1u) {
            let tangent = select(tangents.first, tangents.second, i == 1u);
            let outcome = solve_point_row(
                tangent, anchor_a, anchor_b, 0.0, max_bias,
                mass_first, mass_second, first, second, accumulated[i], forces,
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
            forces = outcome.forces;
        }
        for (var i = 0u; i < 2u; i = i + 1u) {
            let tangent = select(tangents.first, tangents.second, i == 1u);
            let outcome = solve_angular_row(tangent, mass_first, mass_second, first, second, accumulated[2u + i], 0.0, forces);
            first = outcome.first;
            second = outcome.second;
            accumulated[2u + i] = outcome.accumulated;
            forces = outcome.forces;
        }
        let twist_outcome = solve_angular_row(axis, mass_first, mass_second, first, second, accumulated[4], 0.0, forces);
        first = twist_outcome.first;
        second = twist_outcome.second;
        accumulated[4] = twist_outcome.accumulated;
        forces = twist_outcome.forces;
        if ((constraint.flags & CONSTRAINT_HAS_MOTOR) != 0u) {
            let current = dot(anchor_b - anchor_a, axis);
            let outcome = solve_driven_point_row(
                axis, anchor_a, anchor_b, current, constraint.motor_target,
                constraint.motor_stiffness, constraint.motor_damping, constraint.motor_speed,
                constraint.motor_max_force, mass_first, mass_second, first, second, accumulated[6], forces,
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[6] = outcome.accumulated;
            forces = outcome.forces;
        }
        if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
            let separation = dot(anchor_b - anchor_a, axis);
            let above = separation > constraint.limit_max;
            let below = separation < constraint.limit_min;
            if (above) {
                let goal = constraint.limit_max;
                let outcome = solve_point_row(axis, anchor_a, anchor_b, goal, max_bias, mass_first, mass_second, first, second, accumulated[5], forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[5] = min(outcome.accumulated, 0.0);
                forces = outcome.forces;
            } else if (below) {
                let goal = constraint.limit_min;
                let outcome = solve_point_row(axis, anchor_a, anchor_b, goal, max_bias, mass_first, mass_second, first, second, accumulated[5], forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[5] = max(outcome.accumulated, 0.0);
                forces = outcome.forces;
            }
        }
    } else if (constraint.kind == CONSTRAINT_BALL) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = select(vec3f(1.0, 0.0, 0.0), select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), i == 2u), i == 1u);
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, 0.0, max_bias,
                mass_first, mass_second, first, second, accumulated[i], forces,
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
            forces = outcome.forces;
        }
        let hinge = hinge_axis(constraint, first);
        let tangents = make_tangents(hinge);
        if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
            let angle = relative_angle(first, second, hinge);
            let above = angle > constraint.limit_max;
            let below = angle < constraint.limit_min;
            if (above) {
                let error = angle - constraint.limit_max;
                let outcome = limit_row(error, accumulated[3], hinge, mass_first, mass_second, first, second, forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[3] = min(outcome.accumulated, 0.0);
                forces = outcome.forces;
            } else if (below) {
                let error = angle - constraint.limit_min;
                let outcome = limit_row(error, accumulated[3], hinge, mass_first, mass_second, first, second, forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[3] = max(outcome.accumulated, 0.0);
                forces = outcome.forces;
            }
        }
        if ((constraint.flags & CONSTRAINT_HAS_SWING) != 0u) {
            let q_rel = quat_mul(quat_conjugate(first.state.orientation), second.state.orientation);
            let perp = q_rel.xyz - hinge * dot(q_rel.xyz, hinge);
            for (var i = 0u; i < 2u; i = i + 1u) {
                let swing_axis = select(tangents.first, tangents.second, i == 1u);
                let limit = select(constraint.swing_a, constraint.swing_b, i == 1u);
                let swing_angle = 2.0 * atan2(dot(perp, swing_axis), abs(q_rel.w));
                if (abs(swing_angle) > limit) {
                    let error = swing_angle - select(limit, -limit, swing_angle < 0.0);
                    let outcome = limit_row(error, accumulated[4u + i], swing_axis, mass_first, mass_second, first, second, forces);
                    first = outcome.first;
                    second = outcome.second;
                    let capped = min(outcome.accumulated, 0.0);
                    accumulated[4u + i] = select(capped, max(outcome.accumulated, 0.0), swing_angle < 0.0);
                    forces = outcome.forces;
                }
            }
        }
    } else if (constraint.kind == CONSTRAINT_GEAR) {
        let axis_a = normalize(quat_rotate(first.state.orientation, constraint.axis_a));
        let axis_b = normalize(quat_rotate(second.state.orientation, constraint.axis_b));
        let ratio = constraint.gear_ratio;
        let k = dot(axis_a, apply_inverse_inertia(mass_first, axis_a)) * ratio * ratio
            + dot(axis_b, apply_inverse_inertia(mass_second, axis_b));
        if (k > 0.0) {
            let jacobian_speed = ratio * dot(first.state.angular_velocity, axis_a) - dot(second.state.angular_velocity, axis_b);
            let next = accumulated[0] + jacobian_speed / k;
            let applied = next - accumulated[0];
            first.state.angular_velocity = first.state.angular_velocity - apply_inverse_inertia(first, axis_a * ratio * applied);
            second.state.angular_velocity = second.state.angular_velocity + apply_inverse_inertia(second, axis_b * applied);
            accumulated[0] = next;
            forces.angular = forces.angular + abs(applied);
        }
    } else if (constraint.kind == CONSTRAINT_PULLEY) {
        let dir_a = sign_normalize(constraint.pulley_fixed_a - anchor_a);
        let dir_b = sign_normalize(constraint.pulley_fixed_b - anchor_b);
        let length = length(anchor_a - constraint.pulley_fixed_a) + length(anchor_b - constraint.pulley_fixed_b);
        let error = length - constraint.distance;
        let ra = anchor_a - body_com(first);
        let rax = cross(ra, dir_a);
        let rb = anchor_b - body_com(second);
        let rbx = cross(rb, dir_b);
        let k = mass_first.desc.inverse_mass + mass_second.desc.inverse_mass
            + dot(rax, apply_inverse_inertia(mass_first, rax))
            + dot(rbx, apply_inverse_inertia(mass_second, rbx));
        if (k > 0.0) {
            let velocity = relative_velocity(first, first, anchor_a, anchor_a);
            let jacobian_speed = dot(velocity, dir_a)
                + dot(relative_velocity(second, second, anchor_b, anchor_b), dir_b);
            let bias_speed = clamp(error * 0.05 / max(params.dt, 1e-4), -max_bias, max_bias);
            let next = accumulated[0] + (bias_speed - jacobian_speed) / k;
            let applied = next - accumulated[0];
            first.state.velocity = first.state.velocity + dir_a * applied * first.desc.inverse_mass;
            first.state.angular_velocity = first.state.angular_velocity + apply_inverse_inertia(first, cross(anchor_a - body_com(first), dir_a * applied));
            second.state.velocity = second.state.velocity + dir_b * applied * second.desc.inverse_mass;
            second.state.angular_velocity = second.state.angular_velocity + apply_inverse_inertia(second, cross(anchor_b - body_com(second), dir_b * applied));
            accumulated[0] = next;
            forces.linear = forces.linear + abs(applied);
        }
    } else if (constraint.kind == CONSTRAINT_CONE) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = select(vec3f(1.0, 0.0, 0.0), select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), i == 2u), i == 1u);
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, 0.0, max_bias,
                mass_first, mass_second, first, second, accumulated[i], forces,
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
            forces = outcome.forces;
        }
        let cone_axis = normalize(quat_rotate(first.state.orientation, constraint.axis_a));
        let limb = normalize(quat_rotate(second.state.orientation, constraint.axis_b));
        let direction = -limb;
        let cosine = clamp(dot(cone_axis, direction), -1.0, 1.0);
        let angle = acos(cosine);
        if (angle > constraint.cone_angle) {
            let swing_axis = sign_normalize(cross(cone_axis, direction));
            let error = angle - constraint.cone_angle;
            let outcome = limit_row(error, accumulated[3], swing_axis, mass_first, mass_second, first, second, forces);
            first = outcome.first;
            second = outcome.second;
            accumulated[3] = min(outcome.accumulated, 0.0);
            forces = outcome.forces;
        }
    } else if (constraint.kind == CONSTRAINT_SIXDOF) {
        let hinge = hinge_axis(constraint, first);
        let tangents = make_tangents(hinge);
        for (var i = 0u; i < 3u; i = i + 1u) {
            let local_axis = dof_frame(tangents, normalize(constraint.axis_a), i);
            let world_axis = sign_normalize(quat_rotate(first.state.orientation, local_axis));
            let mode = dof_mode(constraint.flags, i);
            if (mode == DOF_FREE) {
                continue;
            }
            let current = dot(anchor_b - anchor_a, world_axis);
            if (mode == DOF_LOCKED) {
                let outcome = solve_point_row(world_axis, anchor_a, anchor_b, 0.0, max_bias, mass_first, mass_second, first, second, accumulated[i], forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[i] = outcome.accumulated;
                forces = outcome.forces;
            } else if (mode == DOF_LIMITED) {
                let min_goal = vec_index(constraint.linear_limit_min, i);
                let max_goal = vec_index(constraint.linear_limit_max, i);
                if (current < min_goal) {
                    let outcome = solve_point_row(world_axis, anchor_a, anchor_b, min_goal, max_bias, mass_first, mass_second, first, second, accumulated[i], forces);
                    first = outcome.first;
                    second = outcome.second;
                    accumulated[i] = min(outcome.accumulated, 0.0);
                    forces = outcome.forces;
                } else if (current > max_goal) {
                    let outcome = solve_point_row(world_axis, anchor_a, anchor_b, max_goal, max_bias, mass_first, mass_second, first, second, accumulated[i], forces);
                    first = outcome.first;
                    second = outcome.second;
                    accumulated[i] = max(outcome.accumulated, 0.0);
                    forces = outcome.forces;
                }
            } else {
                let target_value = vec_index(constraint.linear_motor_target, i);
                let stiffness = vec_index(constraint.linear_motor_stiffness, i);
                let damping = vec_index(constraint.linear_motor_damping, i);
                let max_force = vec_index(constraint.linear_motor_force, i);
                let target_speed = select(0.0, target_value, stiffness <= 0.0);
                let outcome = solve_driven_point_row(world_axis, anchor_a, anchor_b, current, target_value, stiffness, damping, target_speed, max_force, mass_first, mass_second, first, second, accumulated[i], forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[i] = outcome.accumulated;
                forces = outcome.forces;
            }
        }
        let q_rel = quat_mul(quat_conjugate(first.state.orientation), second.state.orientation);
        let deviation = quat_mul(q_rel, quat_conjugate(constraint.reference));
        let error_vector = quat_rotate(quat_conjugate(first.state.orientation), vec3f(2.0 * deviation.x, 2.0 * deviation.y, 2.0 * deviation.z));
        for (var i = 0u; i < 3u; i = i + 1u) {
            let local_axis = dof_frame(tangents, normalize(constraint.axis_a), i);
            let world_axis = sign_normalize(quat_rotate(first.state.orientation, local_axis));
            let mode = dof_mode(constraint.flags, 3u + i);
            if (mode == DOF_FREE) {
                continue;
            }
            let current = dot(error_vector, local_axis);
            if (mode == DOF_LOCKED) {
                let outcome = solve_angular_row(world_axis, mass_first, mass_second, first, second, accumulated[3u + i], -current, forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[3u + i] = outcome.accumulated;
                forces = outcome.forces;
            } else if (mode == DOF_LIMITED) {
                let min_goal = vec_index(constraint.angular_limit_min, i);
                let max_goal = vec_index(constraint.angular_limit_max, i);
                if (current < min_goal) {
                    let outcome = limit_row(current - min_goal, accumulated[3u + i], world_axis, mass_first, mass_second, first, second, forces);
                    first = outcome.first;
                    second = outcome.second;
                    accumulated[3u + i] = min(outcome.accumulated, 0.0);
                    forces = outcome.forces;
                } else if (current > max_goal) {
                    let outcome = limit_row(current - max_goal, accumulated[3u + i], world_axis, mass_first, mass_second, first, second, forces);
                    first = outcome.first;
                    second = outcome.second;
                    accumulated[3u + i] = max(outcome.accumulated, 0.0);
                    forces = outcome.forces;
                }
            } else {
                let target_value = vec_index(constraint.angular_motor_target, i);
                let stiffness = vec_index(constraint.angular_motor_stiffness, i);
                let damping = vec_index(constraint.angular_motor_damping, i);
                let max_force = vec_index(constraint.angular_motor_force, i);
                let target_speed = select(0.0, target_value, stiffness <= 0.0);
                let outcome = solve_driven_angular_row(world_axis, current, target_value, stiffness, damping, target_speed, max_force, mass_first, mass_second, first, second, accumulated[3u + i], forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[3u + i] = outcome.accumulated;
                forces = outcome.forces;
            }
        }
    } else {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = select(vec3f(1.0, 0.0, 0.0), select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), i == 2u), i == 1u);
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, 0.0, max_bias,
                mass_first, mass_second, first, second, accumulated[i], forces,
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
            forces = outcome.forces;
        }
        let basis: array<vec3f, 3> = array(
            vec3f(1.0, 0.0, 0.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0, 0.0, 1.0),
        );
        for (var i = 0u; i < 3u; i = i + 1u) {
            let outcome = solve_angular_row(basis[i], mass_first, mass_second, first, second, accumulated[3u + i], 0.0, forces);
            first = outcome.first;
            second = outcome.second;
            accumulated[3u + i] = outcome.accumulated;
            forces = outcome.forces;
        }
    }
    if ((constraint.flags & CONSTRAINT_HAS_BREAK) != 0u) {
        let linear_limit = constraint.break_force * params.dt;
        let angular_limit = constraint.break_torque * params.dt;
        if ((linear_limit > 0.0 && forces.linear > linear_limit) || (angular_limit > 0.0 && forces.angular > angular_limit)) {
            runtime.broken = 1u;
        }
    }
    runtime.accumulated = accumulated;
    constraint_runtime[constraint_index] = runtime;
    let breach_now = constraint_breach(anchor_a, anchor_b, constraint);
    if (first_sleeping && !second_sleeping && breach_now) {
        atomicOr(&wake_flags[constraint.a], 1u);
    }
    if (second_sleeping && !first_sleeping && breach_now) {
        atomicOr(&wake_flags[constraint.b], 1u);
    }
    store_block_delta(
        slot,
        first.state.velocity - pair.first.state.velocity,
        first.state.angular_velocity - pair.first.state.angular_velocity,
        second.state.velocity - pair.second.state.velocity,
        second.state.angular_velocity - pair.second.state.angular_velocity,
    );
}
