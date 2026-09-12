struct Correction {
    linear: vec3f,
    angular: vec3f,
}

struct CorrectionPair {
    first: Correction,
    second: Correction,
}

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn correction_zero() -> Correction {
    var correction: Correction;
    correction.linear = vec3f(0.0);
    correction.angular = vec3f(0.0);
    return correction;
}

fn correction_holds(correction: Correction) -> bool {
    return dot(correction.linear, correction.linear) > 0.0 || dot(correction.angular, correction.angular) > 0.0;
}

fn pair_zero() -> CorrectionPair {
    var pair: CorrectionPair;
    pair.first = correction_zero();
    pair.second = correction_zero();
    return pair;
}

fn pair_holds(pair: CorrectionPair) -> bool {
    return correction_holds(pair.first) || correction_holds(pair.second);
}

fn store_block_correction(slot: u32, pair: CorrectionPair) {
    block_corrections[slot * 4u] = vec4f(pair.first.linear, 0.0);
    block_corrections[slot * 4u + 1u] = vec4f(pair.first.angular, 0.0);
    block_corrections[slot * 4u + 2u] = vec4f(pair.second.linear, 0.0);
    block_corrections[slot * 4u + 3u] = vec4f(pair.second.angular, 0.0);
}

fn apply_correction(body: ptr<function, Body>, correction: Correction) {
    (*body).state.position = (*body).state.position + correction.linear;
    (*body).state.orientation = normalize(quat_mul(vec4f(correction.angular * 0.5, 1.0), (*body).state.orientation));
}

fn row(
    first: ptr<function, Body>,
    second: ptr<function, Body>,
    total: ptr<function, CorrectionPair>,
    correction: CorrectionPair,
) {
    apply_correction(first, correction.first);
    apply_correction(second, correction.second);
    (*total).first.linear = (*total).first.linear + correction.first.linear;
    (*total).first.angular = (*total).first.angular + correction.first.angular;
    (*total).second.linear = (*total).second.linear + correction.second.linear;
    (*total).second.angular = (*total).second.angular + correction.second.angular;
}

fn point_row(
    axis: vec3f,
    point_a: vec3f,
    point_b: vec3f,
    goal: f32,
    scale: f32,
    first: Body,
    second: Body,
) -> CorrectionPair {
    var pair = pair_zero();
    let k = linear_momentum_mass(first, second);
    if (k > 0.0) {
        let magnitude = scale * (dot(point_b - point_a, axis) - goal) / k;
        pair.first.linear = axis * (magnitude * first.desc.inverse_mass);
        pair.second.linear = axis * (-magnitude * second.desc.inverse_mass);
    }
    return pair;
}

fn angular_row(axis: vec3f, error: f32, scale: f32, first: Body, second: Body) -> CorrectionPair {
    var pair = pair_zero();
    let k = dot(axis, apply_inverse_inertia(first, axis) + apply_inverse_inertia(second, axis));
    if (k > 0.0) {
        let magnitude = scale * error / k;
        pair.first.angular = apply_inverse_inertia(first, axis * magnitude);
        pair.second.angular = apply_inverse_inertia(second, axis * -magnitude);
    }
    return pair;
}

fn local_row(local_axis: vec3f, error: f32, scale: f32, first: Body, second: Body) -> CorrectionPair {
    return angular_row(quat_rotate(first.state.orientation, local_axis), error, scale, first, second);
}

fn local_point_row(
    local_a: vec3f,
    local_b: vec3f,
    axis: vec3f,
    goal: f32,
    scale: f32,
    first: Body,
    second: Body,
) -> CorrectionPair {
    return point_row(
        axis,
        constraint_anchor(first, local_a),
        constraint_anchor(second, local_b),
        goal,
        scale,
        first,
        second,
    );
}

fn solve_contact_correction(contact_index: u32, slot: u32) {
    let contact = contacts[contact_index];
    if (!contact_block_resolves(contact)) {
        store_block_correction(slot, pair_zero());
        return;
    }
    let first_row = collider_owners[contact.a];
    let second_row = collider_owners[contact.b];
    let first_loaded = load_body(first_row);
    let second_loaded = load_body(second_row);
    var first = first_loaded;
    var second = second_loaded;
    if (body_is_inert(first_loaded)) {
        first = body_frozen(first_loaded);
    }
    if (body_is_inert(second_loaded)) {
        second = body_frozen(second_loaded);
    }
    let normal = contact.normal;
    let resolved = dot(normal, resolution[second_row].xyz - resolution[first_row].xyz);
    var total = vec3f(0.0);
    var contributing = 0u;
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        let error = max(contact.points[point_index].depth - params.slop - resolved, 0.0);
        if (error <= 0.0) {
            continue;
        }
        let k = linear_momentum_mass(first, second);
        if (k == 0.0) {
            continue;
        }
        total = total + normal * (params.relaxation * error / k);
        contributing = contributing + 1u;
    }
    if (contributing == 0u) {
        store_block_correction(slot, pair_zero());
        return;
    }
    atomicAdd(&contributions[first_row], 1u);
    atomicAdd(&contributions[second_row], 1u);
    let correction = total / f32(contributing);
    var pair = pair_zero();
    pair.first.linear = -correction * first.desc.inverse_mass;
    pair.second.linear = correction * second.desc.inverse_mass;
    store_block_correction(slot, pair);
}

fn solve_constraint_correction(constraint_index: u32, slot: u32) {
    let constraint = constraint_descs[constraint_index];
    let rows = constraint_rows[constraint_index];
    var total = pair_zero();
    if (constraint_runtime[constraint_index].broken == 0u) {
        let first_loaded = load_body(rows.first_row);
        let second_loaded = load_body(rows.second_row);
        var first = first_loaded;
        var second = second_loaded;
        if (body_is_inert(first_loaded)) {
            first = body_frozen(first_loaded);
        }
        if (body_is_inert(second_loaded)) {
            second = body_frozen(second_loaded);
        }
        let scale = params.relaxation;
        let local_a = constraint.anchor_a;
        let local_b = constraint.anchor_b;
        if (constraint.kind == CONSTRAINT_DISTANCE) {
            if ((constraint.flags & CONSTRAINT_IS_SPRING) == 0u) {
                let anchor_a = constraint_anchor(first, local_a);
                let anchor_b = constraint_anchor(second, local_b);
                row(&first, &second, &total, point_row(sign_normalize(anchor_b - anchor_a), anchor_a, anchor_b, constraint.distance, scale, first, second));
            }
        } else if (constraint.kind == CONSTRAINT_REVOLUTE) {
            for (var axis_index = 0u; axis_index < 3u; axis_index = axis_index + 1u) {
                row(&first, &second, &total, local_point_row(local_a, local_b, orthogonal_axis(axis_index), 0.0, scale, first, second));
            }
            let local_hinge = normalize(constraint.axis_a);
            let error_vector = constraint_relative_error(first, second, constraint.reference);
            for (var axis_index = 0u; axis_index < 2u; axis_index = axis_index + 1u) {
                let local_axis = constraint_dof_axis(constraint_local_frame(local_hinge), local_hinge, axis_index);
                row(&first, &second, &total, local_row(local_axis, dot(error_vector, local_axis), scale, first, second));
            }
            if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
                let angle = constraint_angle(first, second, local_hinge);
                if (angle > constraint.limit_max) {
                    row(&first, &second, &total, local_row(local_hinge, angle - constraint.limit_max, scale, first, second));
                } else if (angle < constraint.limit_min) {
                    row(&first, &second, &total, local_row(local_hinge, angle - constraint.limit_min, scale, first, second));
                }
            }
        } else if (constraint.kind == CONSTRAINT_PRISMATIC) {
            let local_hinge = normalize(constraint.axis_a);
            let tangents = constraint_local_frame(local_hinge);
            let hinge = quat_rotate(first.state.orientation, local_hinge);
            for (var axis_index = 0u; axis_index < 2u; axis_index = axis_index + 1u) {
                let local_axis = constraint_dof_axis(tangents, local_hinge, axis_index);
                row(&first, &second, &total, local_point_row(local_a, local_b, quat_rotate(first.state.orientation, local_axis), 0.0, scale, first, second));
            }
            let error_vector = constraint_relative_error(first, second, constraint.reference);
            for (var axis_index = 0u; axis_index < 3u; axis_index = axis_index + 1u) {
                let local_axis = constraint_dof_axis(tangents, local_hinge, axis_index);
                row(&first, &second, &total, local_row(local_axis, dot(error_vector, local_axis), scale, first, second));
            }
            if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
                let separation = dot(constraint_anchor(second, local_b) - constraint_anchor(first, local_a), hinge);
                if (separation > constraint.limit_max) {
                    row(&first, &second, &total, local_point_row(local_a, local_b, hinge, constraint.limit_max, scale, first, second));
                } else if (separation < constraint.limit_min) {
                    row(&first, &second, &total, local_point_row(local_a, local_b, hinge, constraint.limit_min, scale, first, second));
                }
            }
        } else if (constraint.kind == CONSTRAINT_BALL) {
            for (var axis_index = 0u; axis_index < 3u; axis_index = axis_index + 1u) {
                row(&first, &second, &total, local_point_row(local_a, local_b, orthogonal_axis(axis_index), 0.0, scale, first, second));
            }
            let local_hinge = normalize(constraint.axis_a);
            if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
                let angle = constraint_angle(first, second, local_hinge);
                if (angle > constraint.limit_max) {
                    row(&first, &second, &total, local_row(local_hinge, angle - constraint.limit_max, scale, first, second));
                } else if (angle < constraint.limit_min) {
                    row(&first, &second, &total, local_row(local_hinge, angle - constraint.limit_min, scale, first, second));
                }
            }
            if ((constraint.flags & CONSTRAINT_HAS_SWING) != 0u) {
                let tangents = constraint_local_frame(local_hinge);
                let q_rel = quat_mul(quat_conjugate(first.state.orientation), second.state.orientation);
                let perpendicular = q_rel.xyz - local_hinge * dot(q_rel.xyz, local_hinge);
                for (var axis_index = 0u; axis_index < 2u; axis_index = axis_index + 1u) {
                    let local_axis = constraint_dof_axis(tangents, local_hinge, axis_index);
                    let limit = select(constraint.swing_a, constraint.swing_b, axis_index == 1u);
                    let swing_angle = 2.0 * atan2(dot(perpendicular, local_axis), abs(q_rel.w));
                    if (abs(swing_angle) > limit) {
                        row(&first, &second, &total, local_row(local_axis, swing_angle - select(limit, -limit, swing_angle < 0.0), scale, first, second));
                    }
                }
            }
        } else if (constraint.kind == CONSTRAINT_PULLEY) {
            let anchor_a = constraint_anchor(first, local_a);
            let anchor_b = constraint_anchor(second, local_b);
            let direction_a = sign_normalize(constraint.pulley_fixed_a - anchor_a);
            let direction_b = sign_normalize(constraint.pulley_fixed_b - anchor_b);
            let length_ab = length(anchor_a - constraint.pulley_fixed_a) + length(anchor_b - constraint.pulley_fixed_b);
            let k = linear_momentum_mass(first, second);
            if (k > 0.0) {
                let magnitude = scale * (length_ab - constraint.distance) / k;
                var pulley = pair_zero();
                pulley.first.linear = direction_a * (magnitude * first.desc.inverse_mass);
                pulley.second.linear = direction_b * (magnitude * second.desc.inverse_mass);
                row(&first, &second, &total, pulley);
            }
        } else if (constraint.kind == CONSTRAINT_CONE) {
            for (var axis_index = 0u; axis_index < 3u; axis_index = axis_index + 1u) {
                row(&first, &second, &total, local_point_row(local_a, local_b, orthogonal_axis(axis_index), 0.0, scale, first, second));
            }
            let cone_axis = normalize(quat_rotate(first.state.orientation, constraint.axis_a));
            let limb = normalize(quat_rotate(second.state.orientation, constraint.axis_b));
            let direction = -limb;
            let angle = acos(clamp(dot(cone_axis, direction), -1.0, 1.0));
            if (angle > constraint.cone_angle) {
                row(&first, &second, &total, angular_row(sign_normalize(cross(cone_axis, direction)), angle - constraint.cone_angle, scale, first, second));
            }
        } else if (constraint.kind == CONSTRAINT_SIXDOF) {
            let local_hinge = normalize(constraint.axis_a);
            let tangents = constraint_local_frame(local_hinge);
            let error_vector = constraint_relative_error(first, second, constraint.reference);
            for (var axis_index = 0u; axis_index < 3u; axis_index = axis_index + 1u) {
                let local_axis = constraint_dof_axis(tangents, local_hinge, axis_index);
                let current = dot(constraint_anchor(second, local_b) - constraint_anchor(first, local_a), quat_rotate(first.state.orientation, local_axis));
                if (dof_locked(constraint.flags, axis_index)) {
                    row(&first, &second, &total, local_point_row(local_a, local_b, quat_rotate(first.state.orientation, local_axis), 0.0, scale, first, second));
                } else if (dof_limited(constraint.flags, axis_index)) {
                    let min_goal = vec_index(constraint.linear_limit_min, axis_index);
                    let max_goal = vec_index(constraint.linear_limit_max, axis_index);
                    if (current < min_goal) {
                        row(&first, &second, &total, local_point_row(local_a, local_b, quat_rotate(first.state.orientation, local_axis), min_goal, scale, first, second));
                    } else if (current > max_goal) {
                        row(&first, &second, &total, local_point_row(local_a, local_b, quat_rotate(first.state.orientation, local_axis), max_goal, scale, first, second));
                    }
                }
                let angular = dot(error_vector, local_axis);
                let row_index = 3u + axis_index;
                if (dof_locked(constraint.flags, row_index)) {
                    row(&first, &second, &total, local_row(local_axis, angular, scale, first, second));
                } else if (dof_limited(constraint.flags, row_index)) {
                    let min_goal = vec_index(constraint.angular_limit_min, axis_index);
                    let max_goal = vec_index(constraint.angular_limit_max, axis_index);
                    if (angular < min_goal) {
                        row(&first, &second, &total, local_row(local_axis, angular - min_goal, scale, first, second));
                    } else if (angular > max_goal) {
                        row(&first, &second, &total, local_row(local_axis, angular - max_goal, scale, first, second));
                    }
                }
            }
        } else if (constraint.kind == CONSTRAINT_FIXED) {
            for (var axis_index = 0u; axis_index < 3u; axis_index = axis_index + 1u) {
                row(&first, &second, &total, local_point_row(local_a, local_b, orthogonal_axis(axis_index), 0.0, scale, first, second));
            }
            let error_vector = constraint_relative_error(first, second, constraint.reference);
            for (var axis_index = 0u; axis_index < 3u; axis_index = axis_index + 1u) {
                row(&first, &second, &total, local_row(orthogonal_axis(axis_index), dot(error_vector, orthogonal_axis(axis_index)), scale, first, second));
            }
        }
        if (pair_holds(total)) {
            atomicAdd(&contributions[rows.first_row], 1u);
            atomicAdd(&contributions[rows.second_row], 1u);
        }
    }
    store_block_correction(slot, total);
}
