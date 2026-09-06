@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read_write> constraints: array<Constraint>;
@group(0) @binding(3) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> bucket_values: array<u32>;

struct RowOutcome {
    first: RigidBody,
    second: RigidBody,
    accumulated: f32,
}

fn constraint_anchor(body: RigidBody, local: vec3f) -> vec3f {
    return body.position + quat_rotate(body.orientation, local);
}

fn constraint_breach(anchor_a: vec3f, anchor_b: vec3f, constraint: Constraint) -> bool {
    let separation = length(anchor_b - anchor_a);
    return abs(separation - constraint.distance) > 0.05;
}

fn solve_point_row(
    axis: vec3f,
    point_a: vec3f,
    point_b: vec3f,
    goal: f32,
    max_bias: f32,
    first: RigidBody,
    second: RigidBody,
    accumulated: f32,
) -> RowOutcome {
    let bias = dot(point_b - point_a, axis) - goal;
    let k = point_momentum_mass(first, second, point_a, point_b, axis);
    var first_out = first;
    var second_out = second;
    var accumulated_out = accumulated;
    if (k > 0.0) {
        let velocity = relative_velocity(first, second, point_a, point_b);
        let jacobian_speed = dot(velocity, axis);
        let bias_speed = clamp(bias * 0.05 / max(params.dt, 1e-4), -max_bias, max_bias);
        let next = accumulated - (jacobian_speed + bias_speed) / k;
        let applied = next - accumulated;
        apply_pair_impulse(&first_out, &second_out, point_a, point_b, axis * applied);
        accumulated_out = next;
    }
    return RowOutcome(first_out, second_out, accumulated_out);
}

fn solve_angular_row(
    axis: vec3f,
    first: RigidBody,
    second: RigidBody,
    accumulated: f32,
    bias_speed: f32,
) -> RowOutcome {
    let axis_normalized = normalize(axis);
    let k = dot(
        axis_normalized,
        apply_inverse_inertia(first, axis_normalized) + apply_inverse_inertia(second, axis_normalized),
    );
    var first_out = first;
    var second_out = second;
    var accumulated_out = accumulated;
    if (k > 0.0) {
        let velocity = second.angular_velocity - first.angular_velocity;
        let jacobian_speed = dot(velocity, axis_normalized);
        let next = accumulated - (jacobian_speed + bias_speed) / k;
        let applied = next - accumulated;
        first_out.angular_velocity =
            first.angular_velocity - apply_inverse_inertia(first, axis_normalized * applied);
        second_out.angular_velocity =
            second.angular_velocity + apply_inverse_inertia(second, axis_normalized * applied);
        accumulated_out = next;
    }
    return RowOutcome(first_out, second_out, accumulated_out);
}

fn hinge_axis(constraint: Constraint, first: RigidBody) -> vec3f {
    return normalize(quat_rotate(first.orientation, constraint.axis_a));
}

fn relative_angle(first: RigidBody, second: RigidBody, hinge: vec3f) -> f32 {
    let q_rel = quat_mul(quat_conjugate(first.orientation), second.orientation);
    let signed = atan2(dot(q_rel.xyz, hinge), q_rel.w) * 2.0;
    return signed;
}

fn limit_row(
    error: f32,
    accumulated: f32,
    axis: vec3f,
    first: RigidBody,
    second: RigidBody,
) -> RowOutcome {
    let bias_speed = clamp(error * 0.05 / max(params.dt, 1e-4), -4.0, 4.0);
    let outcome = solve_angular_row(axis, first, second, accumulated, bias_speed);
    return outcome;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.constraint_count) {
        return;
    }
    let constraint_index = bucket_values[index];
    var constraint = constraints[constraint_index];
    if (constraint.kind == CONSTRAINT_INVALID) {
        return;
    }
    var first = bodies[constraint.a];
    var second = bodies[constraint.b];
    let first_sleeping = (first.flags & BODY_SLEEPING) != 0u;
    let second_sleeping = (second.flags & BODY_SLEEPING) != 0u;
    if (body_is_inert(first)) {
        first = body_frozen(first);
    }
    if (body_is_inert(second)) {
        second = body_frozen(second);
    }
    let anchor_a = constraint_anchor(first, constraint.anchor_a);
    let anchor_b = constraint_anchor(second, constraint.anchor_b);
    var accumulated = constraint.accumulated;
    let max_bias = 4.0;
    if (constraint.kind == CONSTRAINT_DISTANCE) {
        let axis = sign_normalize(anchor_b - anchor_a);
        if ((constraint.flags & CONSTRAINT_IS_SPRING) == 0u) {
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, constraint.distance, max_bias,
                first, second, accumulated[0],
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[0] = outcome.accumulated;
        } else {
            let separation = length(anchor_b - anchor_a);
            let error = separation - constraint.distance;
            let omega = 6.2831853 * constraint.spring_frequency;
            let zeta = constraint.spring_damping_ratio;
            let velocity = relative_velocity(first, second, anchor_a, anchor_b);
            let jacobian_speed = dot(velocity, axis);
            let k = point_momentum_mass(first, second, anchor_a, anchor_b, axis);
            if (k > 0.0) {
                let v_target = -(omega * omega * error * params.dt) - 2.0 * zeta * omega * params.dt * jacobian_speed;
                let next = accumulated[0] - (jacobian_speed - v_target) / k;
                let applied = next - accumulated[0];
                apply_pair_impulse(&first, &second, anchor_a, anchor_b, axis * applied);
                accumulated[0] = next;
            }
        }
    } else if (constraint.kind == CONSTRAINT_REVOLUTE) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = select(vec3f(1.0, 0.0, 0.0), select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), i == 2u), i == 1u);
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, 0.0, max_bias,
                first, second, accumulated[i],
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
        }
        let hinge = hinge_axis(constraint, first);
        let tangents = make_tangents(hinge);
        for (var i = 0u; i < 2u; i = i + 1u) {
            let axis = select(tangents.first, tangents.second, i == 1u);
            let outcome = solve_angular_row(axis, first, second, accumulated[3u + i], 0.0);
            first = outcome.first;
            second = outcome.second;
            accumulated[3u + i] = outcome.accumulated;
        }
        if ((constraint.flags & CONSTRAINT_HAS_MOTOR) != 0u) {
            let outcome = solve_angular_row(hinge, first, second, accumulated[6], -constraint.motor_speed);
            first = outcome.first;
            second = outcome.second;
            accumulated[6] = outcome.accumulated;
        }
        if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
            let angle = relative_angle(first, second, hinge);
            let above = angle > constraint.limit_max;
            let below = angle < constraint.limit_min;
            if (above) {
                let error = angle - constraint.limit_max;
                let outcome = limit_row(error, accumulated[5], hinge, first, second);
                first = outcome.first;
                second = outcome.second;
                accumulated[5] = min(outcome.accumulated, 0.0);
            } else if (below) {
                let error = angle - constraint.limit_min;
                let outcome = limit_row(error, accumulated[5], hinge, first, second);
                first = outcome.first;
                second = outcome.second;
                accumulated[5] = max(outcome.accumulated, 0.0);
            }
        }
    } else if (constraint.kind == CONSTRAINT_PRISMATIC) {
        let axis = normalize(quat_rotate(first.orientation, constraint.axis_a));
        let tangents = make_tangents(axis);
        for (var i = 0u; i < 2u; i = i + 1u) {
            let tangent = select(tangents.first, tangents.second, i == 1u);
            let outcome = solve_point_row(
                tangent, anchor_a, anchor_b, 0.0, max_bias,
                first, second, accumulated[i],
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
        }
        for (var i = 0u; i < 2u; i = i + 1u) {
            let tangent = select(tangents.first, tangents.second, i == 1u);
            let outcome = solve_angular_row(tangent, first, second, accumulated[2u + i], 0.0);
            first = outcome.first;
            second = outcome.second;
            accumulated[2u + i] = outcome.accumulated;
        }
        if ((constraint.flags & CONSTRAINT_HAS_MOTOR) != 0u) {
            let velocity = relative_velocity(first, second, anchor_a, anchor_b);
            let jacobian_speed = dot(velocity, axis);
            let k = point_momentum_mass(first, second, anchor_a, anchor_b, axis);
            if (k > 0.0) {
                let next = accumulated[5] - (jacobian_speed - constraint.motor_speed) / k;
                let applied = next - accumulated[5];
                apply_pair_impulse(&first, &second, anchor_a, anchor_b, axis * applied);
                accumulated[5] = next;
            }
        }
        if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
            let separation = dot(anchor_b - anchor_a, axis);
            let above = separation > constraint.limit_max;
            let below = separation < constraint.limit_min;
            if (above) {
                let goal = constraint.limit_max;
                let outcome = solve_point_row(axis, anchor_a, anchor_b, goal, max_bias, first, second, accumulated[4]);
                first = outcome.first;
                second = outcome.second;
                accumulated[4] = min(outcome.accumulated, 0.0);
            } else if (below) {
                let goal = constraint.limit_min;
                let outcome = solve_point_row(axis, anchor_a, anchor_b, goal, max_bias, first, second, accumulated[4]);
                first = outcome.first;
                second = outcome.second;
                accumulated[4] = max(outcome.accumulated, 0.0);
            }
        }
    } else if (constraint.kind == CONSTRAINT_BALL) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = select(vec3f(1.0, 0.0, 0.0), select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), i == 2u), i == 1u);
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, 0.0, max_bias,
                first, second, accumulated[i],
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
        }
    } else {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = select(vec3f(1.0, 0.0, 0.0), select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), i == 2u), i == 1u);
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, 0.0, max_bias,
                first, second, accumulated[i],
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
        }
        let basis: array<vec3f, 3> = array(
            vec3f(1.0, 0.0, 0.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0, 0.0, 1.0),
        );
        for (var i = 0u; i < 3u; i = i + 1u) {
            let outcome = solve_angular_row(basis[i], first, second, accumulated[3u + i], 0.0);
            first = outcome.first;
            second = outcome.second;
            accumulated[3u + i] = outcome.accumulated;
        }
    }
    constraint.accumulated = accumulated;
    constraints[constraint_index] = constraint;
    let breach_now = constraint_breach(anchor_a, anchor_b, constraint);
    if (first_sleeping && !second_sleeping && breach_now) {
        atomicOr(&wake_flags[constraint.a], 1u);
    }
    if (second_sleeping && !first_sleeping && breach_now) {
        atomicOr(&wake_flags[constraint.b], 1u);
    }
    first.inverse_mass = bodies[constraint.a].inverse_mass;
    first.inverse_inertia_body = bodies[constraint.a].inverse_inertia_body;
    second.inverse_mass = bodies[constraint.b].inverse_mass;
    second.inverse_inertia_body = bodies[constraint.b].inverse_inertia_body;
    bodies[constraint.a] = first;
    bodies[constraint.b] = second;
}
