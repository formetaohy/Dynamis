@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read_write> constraints: array<Constraint>;

const MAX_CONSTRAINT_ROWS: u32 = 6u;

struct PointRow {
    axis: vec3f,
    point_a: vec3f,
    point_b: vec3f,
    goal: f32,
}

struct RowOutcome {
    first: RigidBody,
    second: RigidBody,
    accumulated: f32,
}

fn constraint_anchor(body: RigidBody, local: vec3f) -> vec3f {
    return body.position + quat_rotate(body.orientation, local);
}

fn solve_point_row(
    row: PointRow,
    first: RigidBody,
    second: RigidBody,
    accumulated: f32,
) -> RowOutcome {
    let axis = row.axis;
    let point_a = row.point_a;
    let point_b = row.point_b;
    let bias = dot(point_b - point_a, axis) - row.goal;
    let k = point_momentum_mass(first, second, point_a, point_b, axis);
    var first_out = first;
    var second_out = second;
    var accumulated_out = accumulated;
    if (k > 0.0) {
        let velocity = relative_velocity(first, second, point_a, point_b);
        let jacobian_speed = dot(velocity, axis);
        let bias_speed = clamp(bias * 0.05 / max(params.dt, 1e-4), -4.0, 4.0);
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
) -> RowOutcome {
    let k = dot(
        axis,
        apply_inverse_inertia(first, axis) + apply_inverse_inertia(second, axis),
    );
    var first_out = first;
    var second_out = second;
    var accumulated_out = accumulated;
    if (k > 0.0) {
        let velocity = second.angular_velocity - first.angular_velocity;
        let jacobian_speed = dot(velocity, axis);
        let next = accumulated - jacobian_speed / k;
        let applied = next - accumulated;
        first_out.angular_velocity =
            first.angular_velocity - apply_inverse_inertia(first, axis * applied);
        second_out.angular_velocity =
            second.angular_velocity + apply_inverse_inertia(second, axis * applied);
        accumulated_out = next;
    }
    return RowOutcome(first_out, second_out, accumulated_out);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.constraint_count) {
        return;
    }
    var constraint = constraints[index];
    if (constraint.kind == CONSTRAINT_INVALID) {
        return;
    }
    var first = bodies[constraint.a];
    var second = bodies[constraint.b];
    let anchor_a = constraint_anchor(first, constraint.anchor_a);
    let anchor_b = constraint_anchor(second, constraint.anchor_b);
    var accumulated = constraint.accumulated;
    var row_count = 0u;
    if (constraint.kind == CONSTRAINT_DISTANCE) {
        let axis = sign_normalize(anchor_b - anchor_a);
        let outcome = solve_point_row(
            PointRow(axis, anchor_a, anchor_b, constraint.distance),
            first,
            second,
            accumulated[0],
        );
        first = outcome.first;
        second = outcome.second;
        accumulated[0] = outcome.accumulated;
        row_count = 1u;
    } else if (constraint.kind == CONSTRAINT_REVOLUTE) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = select(vec3f(1.0, 0.0, 0.0), select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), i == 2u), i == 1u);
            let outcome = solve_point_row(
                PointRow(axis, anchor_a, anchor_b, 0.0),
                first,
                second,
                accumulated[i],
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
        }
        let hinge = normalize(quat_rotate(first.orientation, constraint.axis_a));
        let tangents = make_tangents(hinge);
        for (var i = 0u; i < 2u; i = i + 1u) {
            let axis = select(tangents.first, tangents.second, i == 1u);
            let outcome = solve_angular_row(axis, first, second, accumulated[3u + i]);
            first = outcome.first;
            second = outcome.second;
            accumulated[3u + i] = outcome.accumulated;
        }
        row_count = 5u;
    } else if (constraint.kind == CONSTRAINT_PRISMATIC) {
        let axis = normalize(quat_rotate(first.orientation, constraint.axis_a));
        let tangents = make_tangents(axis);
        for (var i = 0u; i < 2u; i = i + 1u) {
            let tangent = select(tangents.first, tangents.second, i == 1u);
            let outcome = solve_point_row(
                PointRow(tangent, anchor_a, anchor_b, 0.0),
                first,
                second,
                accumulated[i],
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
        }
        for (var i = 0u; i < 2u; i = i + 1u) {
            let tangent = select(tangents.first, tangents.second, i == 1u);
            let outcome = solve_angular_row(tangent, first, second, accumulated[2u + i]);
            first = outcome.first;
            second = outcome.second;
            accumulated[2u + i] = outcome.accumulated;
        }
        row_count = 4u;
    } else if (constraint.kind == CONSTRAINT_BALL) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = select(vec3f(1.0, 0.0, 0.0), select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), i == 2u), i == 1u);
            let outcome = solve_point_row(
                PointRow(axis, anchor_a, anchor_b, 0.0),
                first,
                second,
                accumulated[i],
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
        }
        row_count = 3u;
    } else {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = select(vec3f(1.0, 0.0, 0.0), select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), i == 2u), i == 1u);
            let outcome = solve_point_row(
                PointRow(axis, anchor_a, anchor_b, 0.0),
                first,
                second,
                accumulated[i],
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
            let outcome = solve_angular_row(basis[i], first, second, accumulated[3u + i]);
            first = outcome.first;
            second = outcome.second;
            accumulated[3u + i] = outcome.accumulated;
        }
        row_count = 6u;
    }
    constraint.accumulated = accumulated;
    constraints[index] = constraint;
    bodies[constraint.a] = first;
    bodies[constraint.b] = second;
}
