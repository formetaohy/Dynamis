struct FrameForces {
    linear_first: vec3f,
    angular_first: vec3f,
    linear_second: vec3f,
    angular_second: vec3f,
}

fn frame_forces_zero() -> FrameForces {
    var forces: FrameForces;
    forces.linear_first = vec3f(0.0);
    forces.angular_first = vec3f(0.0);
    forces.linear_second = vec3f(0.0);
    forces.angular_second = vec3f(0.0);
    return forces;
}

fn reaction_of(forces: FrameForces) -> ConstraintReaction {
    var reaction: ConstraintReaction;
    reaction.linear_first = forces.linear_first;
    reaction._pad_linear_first = 0.0;
    reaction.angular_first = forces.angular_first;
    reaction._pad_angular_first = 0.0;
    reaction.linear_second = forces.linear_second;
    reaction._pad_linear_second = 0.0;
    reaction.angular_second = forces.angular_second;
    reaction._pad_angular_second = 0.0;
    return reaction;
}

fn accumulate_reaction(
    forces: FrameForces,
    first: Body,
    second: Body,
    point_a: vec3f,
    point_b: vec3f,
    impulse_first: vec3f,
    impulse_second: vec3f,
) -> FrameForces {
    var forces_out = forces;
    forces_out.linear_first = forces_out.linear_first + impulse_first;
    forces_out.linear_second = forces_out.linear_second + impulse_second;
    forces_out.angular_first = forces_out.angular_first + cross(point_a - body_com(first), impulse_first);
    forces_out.angular_second = forces_out.angular_second + cross(point_b - body_com(second), impulse_second);
    return forces_out;
}

fn accumulate_angular_reaction(
    forces: FrameForces,
    couple_first: vec3f,
    couple_second: vec3f,
) -> FrameForces {
    var forces_out = forces;
    forces_out.angular_first = forces_out.angular_first + couple_first;
    forces_out.angular_second = forces_out.angular_second + couple_second;
    return forces_out;
}

fn point_reaction(
    forces: FrameForces,
    first: Body,
    second: Body,
    point_a: vec3f,
    point_b: vec3f,
    axis: vec3f,
    accumulated: f32,
) -> FrameForces {
    let impulse = axis * accumulated;
    return accumulate_reaction(forces, first, second, point_a, point_b, -impulse, impulse);
}

fn angular_reaction(forces: FrameForces, axis: vec3f, accumulated: f32) -> FrameForces {
    let impulse = axis * accumulated;
    return accumulate_angular_reaction(forces, -impulse, impulse);
}

struct RowOutcome {
    first: Body,
    second: Body,
    accumulated: f32,
    forces: FrameForces,
}

struct RowImpulse {
    applied: f32,
    accumulated: f32,
}

fn row_impulse(accumulated: f32, next: f32, cap: f32) -> RowImpulse {
    var total = next;
    if (cap > 0.0) {
        total = clamp(next, -cap, cap);
    }
    return RowImpulse(total - accumulated, total);
}

fn solve_point_row(
    axis: vec3f,
    point_a: vec3f,
    point_b: vec3f,
    goal: f32,
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
        let next = accumulated - jacobian_speed / k;
        let impulse = row_impulse(accumulated, next, 0.0);
        apply_pair_impulse(&first_out, &second_out, point_a, point_b, axis * impulse.applied);
        accumulated_out = impulse.accumulated;
        forces_out = point_reaction(forces_out, first, second, point_a, point_b, axis, accumulated_out);
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
        let next = accumulated - jacobian_speed / k;
        let impulse = row_impulse(accumulated, next, 0.0);
        first_out.state.angular_velocity =
            first.state.angular_velocity - apply_inverse_inertia(first, axis_normalized * impulse.applied);
        second_out.state.angular_velocity =
            second.state.angular_velocity + apply_inverse_inertia(second, axis_normalized * impulse.applied);
        accumulated_out = impulse.accumulated;
        forces_out = angular_reaction(forces_out, axis_normalized, accumulated_out);
    }
    var outcome: RowOutcome;
    outcome.first = first_out;
    outcome.second = second_out;
    outcome.accumulated = accumulated_out;
    outcome.forces = forces_out;
    return outcome;
}

fn solve_drive_row(
    drive: JointDrive,
    mass_first: Body,
    mass_second: Body,
    first: Body,
    second: Body,
    accumulated: f32,
    forces: FrameForces,
) -> RowOutcome {
    let axis = normalize(drive.axis);
    let angular_mass = dot(
        axis,
        apply_inverse_inertia(mass_first, axis) + apply_inverse_inertia(mass_second, axis),
    );
    let point_mass = point_momentum_mass(mass_first, mass_second, drive.point_a, drive.point_b, axis);
    let k = select(point_mass, angular_mass, drive.angular);
    var first_out = first;
    var second_out = second;
    var accumulated_out = accumulated;
    var forces_out = forces;
    if (k > 0.0) {
        let next = accumulated - (drive.rate - drive.target_speed) / k;
        let impulse = row_impulse(accumulated, next, drive.max_force * params.dt);
        if (drive.angular) {
            first_out.state.angular_velocity =
                first.state.angular_velocity - apply_inverse_inertia(first, axis * impulse.applied);
            second_out.state.angular_velocity =
                second.state.angular_velocity + apply_inverse_inertia(second, axis * impulse.applied);
            accumulated_out = impulse.accumulated;
            forces_out = angular_reaction(forces_out, axis, accumulated_out);
        } else {
            apply_pair_impulse(&first_out, &second_out, drive.point_a, drive.point_b, axis * impulse.applied);
            accumulated_out = impulse.accumulated;
            forces_out = point_reaction(
                forces_out,
                first,
                second,
                drive.point_a,
                drive.point_b,
                axis,
                accumulated_out,
            );
        }
    }
    var outcome: RowOutcome;
    outcome.first = first_out;
    outcome.second = second_out;
    outcome.accumulated = accumulated_out;
    outcome.forces = forces_out;
    return outcome;
}

fn limit_row(
    accumulated: f32,
    axis: vec3f,
    mass_first: Body,
    mass_second: Body,
    first: Body,
    second: Body,
    forces: FrameForces,
) -> RowOutcome {
    return solve_angular_row(axis, mass_first, mass_second, first, second, accumulated, forces);
}

fn dof_limit_row(index: u32) -> u32 {
    return DOF_LIMIT_ROW_BASE + index;
}

fn solve_joint(constraint_index: u32, residual: bool) {
    var runtime = constraint_runtime[constraint_index];
    let constraint = constraint_descs[constraint_index];
    let reference = runtime.reference;
    let rows = constraint_rows[constraint_index];
    if ((constraint.flags & CONSTRAINT_WARM_START) == 0u) {
        for (var index = 0u; index < CONSTRAINT_ACCUMULATOR_SLOTS; index = index + 1u) {
            runtime.accumulated[index] = 0.0;
        }
    }
    if (runtime.broken != 0u) {
        return;
    }
    let first_loaded = load_body(rows.first_row);
    let second_loaded = load_body(rows.second_row);
    if (body_is_inert(first_loaded) && body_is_inert(second_loaded)) {
        return;
    }
    var first = first_loaded;
    var second = second_loaded;
    if (body_is_inert(first_loaded)) {
        first = body_frozen(first_loaded);
    }
    if (body_is_inert(second_loaded)) {
        second = body_frozen(second_loaded);
    }
    let mass_first = first;
    let mass_second = second;
    let anchor_a = constraint_anchor(first, constraint.anchor_a);
    let anchor_b = constraint_anchor(second, constraint.anchor_b);
    var accumulated = runtime.accumulated;
    var forces = frame_forces_zero();
    if (constraint.kind == CONSTRAINT_DISTANCE) {
        let axis = sign_normalize(anchor_b - anchor_a);
        if ((constraint.flags & CONSTRAINT_IS_SPRING) == 0u) {
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, constraint.distance,
                mass_first, mass_second, first, second, accumulated[0], forces,
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[0] = outcome.accumulated;
            forces = outcome.forces;
        } else {
            let omega = 6.2831853 * constraint.spring_frequency;
            let stiffness = omega * omega;
            let damping = 2.0 * constraint.spring_damping_ratio * omega;
            let separation = joint_coordinate(
                constraint,
                reference,
                first,
                second,
                anchor_a,
                anchor_b,
                0u,
            );
            let error = separation - constraint.distance;
            let velocity = relative_velocity(first, second, anchor_a, anchor_b);
            let jacobian_speed = dot(velocity, axis);
            let impulse = -(stiffness * error + damping * jacobian_speed) * params.dt;
            apply_pair_impulse(&first, &second, anchor_a, anchor_b, axis * impulse);
            accumulated[0] = accumulated[0] + impulse;
            forces = point_reaction(forces, first, second, anchor_a, anchor_b, axis, accumulated[0]);
        }
    } else if (constraint.kind == CONSTRAINT_REVOLUTE) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = orthogonal_axis(i);
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, 0.0,
                mass_first, mass_second, first, second, accumulated[i], forces,
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
            forces = outcome.forces;
        }
        let local_hinge = normalize(constraint.axis_a);
        let tangents = constraint_local_frame(local_hinge);
        for (var i = 0u; i < 2u; i = i + 1u) {
            let local_axis = select(tangents.first, tangents.second, i == 1u);
            let axis = quat_rotate(first.state.orientation, local_axis);
            let outcome = solve_angular_row(axis, mass_first, mass_second, first, second, accumulated[3u + i], forces);
            first = outcome.first;
            second = outcome.second;
            accumulated[3u + i] = outcome.accumulated;
            forces = outcome.forces;
        }
        let hinge = quat_rotate(first.state.orientation, local_hinge);
        if ((constraint.flags & CONSTRAINT_HAS_MOTOR) != 0u) {
            let drive = scalar_drive(constraint, reference, first, second, anchor_a, anchor_b, true, params.dt);
            let outcome = solve_drive_row(drive, mass_first, mass_second, first, second, accumulated[6], forces);
            first = outcome.first;
            second = outcome.second;
            accumulated[6] = outcome.accumulated;
            forces = outcome.forces;
        }
        if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
            let angle = joint_coordinate(
                constraint,
                reference,
                first,
                second,
                anchor_a,
                anchor_b,
                0u,
            );
            if (angle > constraint.limit_max) {
                let outcome = limit_row(accumulated[5], hinge, mass_first, mass_second, first, second, forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[5] = min(outcome.accumulated, 0.0);
                forces = outcome.forces;
            } else if (angle < constraint.limit_min) {
                let outcome = limit_row(accumulated[5], hinge, mass_first, mass_second, first, second, forces);
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
                tangent, anchor_a, anchor_b, 0.0,
                mass_first, mass_second, first, second, accumulated[i], forces,
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
            forces = outcome.forces;
        }
        for (var i = 0u; i < 2u; i = i + 1u) {
            let tangent = select(tangents.first, tangents.second, i == 1u);
            let outcome = solve_angular_row(tangent, mass_first, mass_second, first, second, accumulated[2u + i], forces);
            first = outcome.first;
            second = outcome.second;
            accumulated[2u + i] = outcome.accumulated;
            forces = outcome.forces;
        }
        let twist_outcome = solve_angular_row(axis, mass_first, mass_second, first, second, accumulated[4], forces);
        first = twist_outcome.first;
        second = twist_outcome.second;
        accumulated[4] = twist_outcome.accumulated;
        forces = twist_outcome.forces;
        if ((constraint.flags & CONSTRAINT_HAS_MOTOR) != 0u) {
            let drive = scalar_drive(constraint, reference, first, second, anchor_a, anchor_b, false, params.dt);
            let outcome = solve_drive_row(drive, mass_first, mass_second, first, second, accumulated[6], forces);
            first = outcome.first;
            second = outcome.second;
            accumulated[6] = outcome.accumulated;
            forces = outcome.forces;
        }
        if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
            let separation = joint_coordinate(
                constraint,
                reference,
                first,
                second,
                anchor_a,
                anchor_b,
                0u,
            );
            let above = separation > constraint.limit_max;
            let below = separation < constraint.limit_min;
            if (above) {
                let goal = constraint.limit_max;
                let outcome = solve_point_row(axis, anchor_a, anchor_b, goal, mass_first, mass_second, first, second, accumulated[5], forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[5] = min(outcome.accumulated, 0.0);
                forces = outcome.forces;
            } else if (below) {
                let goal = constraint.limit_min;
                let outcome = solve_point_row(axis, anchor_a, anchor_b, goal, mass_first, mass_second, first, second, accumulated[5], forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[5] = max(outcome.accumulated, 0.0);
                forces = outcome.forces;
            }
        }
    } else if (constraint.kind == CONSTRAINT_BALL) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = orthogonal_axis(i);
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, 0.0,
                mass_first, mass_second, first, second, accumulated[i], forces,
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
            forces = outcome.forces;
        }
        let local_hinge = normalize(constraint.axis_a);
        let hinge = quat_rotate(first.state.orientation, local_hinge);
        if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
            let angle = joint_coordinate(
                constraint,
                reference,
                first,
                second,
                anchor_a,
                anchor_b,
                2u,
            );
            if (angle > constraint.limit_max) {
                let outcome = limit_row(accumulated[3], hinge, mass_first, mass_second, first, second, forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[3] = min(outcome.accumulated, 0.0);
                forces = outcome.forces;
            } else if (angle < constraint.limit_min) {
                let outcome = limit_row(accumulated[3], hinge, mass_first, mass_second, first, second, forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[3] = max(outcome.accumulated, 0.0);
                forces = outcome.forces;
            }
        }
        if ((constraint.flags & CONSTRAINT_HAS_SWING) != 0u) {
            for (var i = 0u; i < 2u; i = i + 1u) {
                let swing_axis = joint_axis(constraint, first, second, anchor_a, anchor_b, i);
                let limit = select(constraint.swing_a, constraint.swing_b, i == 1u);
                let swing_angle = joint_coordinate(
                    constraint,
                    reference,
                    first,
                    second,
                    anchor_a,
                    anchor_b,
                    i,
                );
                if (abs(swing_angle) > limit) {
                    let outcome = limit_row(accumulated[4u + i], swing_axis, mass_first, mass_second, first, second, forces);
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
            let impulse = row_impulse(accumulated[0], next, 0.0);
            first.state.angular_velocity = first.state.angular_velocity - apply_inverse_inertia(first, axis_a * ratio * impulse.applied);
            second.state.angular_velocity = second.state.angular_velocity + apply_inverse_inertia(second, axis_b * impulse.applied);
            accumulated[0] = impulse.accumulated;
            forces = accumulate_angular_reaction(
                forces,
                axis_a * ratio * accumulated[0],
                axis_b * accumulated[0],
            );
        }
    } else if (constraint.kind == CONSTRAINT_PULLEY) {
        let dir_a = sign_normalize(constraint.pulley_fixed_a - anchor_a);
        let dir_b = sign_normalize(constraint.pulley_fixed_b - anchor_b);
        let ra = anchor_a - body_com(first);
        let rax = cross(ra, dir_a);
        let rb = anchor_b - body_com(second);
        let rbx = cross(rb, dir_b);
        let k = mass_first.desc.inverse_mass + mass_second.desc.inverse_mass
            + dot(rax, apply_inverse_inertia(mass_first, rax))
            + dot(rbx, apply_inverse_inertia(mass_second, rbx));
        if (k > 0.0) {
            let jacobian_speed = dot(point_velocity(first, anchor_a), dir_a)
                + dot(point_velocity(second, anchor_b), dir_b);
            let next = accumulated[0] - jacobian_speed / k;
            let impulse = row_impulse(accumulated[0], next, 0.0);
            first.state.velocity = first.state.velocity + dir_a * impulse.applied * first.desc.inverse_mass;
            first.state.angular_velocity = first.state.angular_velocity + apply_inverse_inertia(first, cross(anchor_a - body_com(first), dir_a * impulse.applied));
            second.state.velocity = second.state.velocity + dir_b * impulse.applied * second.desc.inverse_mass;
            second.state.angular_velocity = second.state.angular_velocity + apply_inverse_inertia(second, cross(anchor_b - body_com(second), dir_b * impulse.applied));
            accumulated[0] = impulse.accumulated;
            forces = accumulate_reaction(
                forces,
                first,
                second,
                anchor_a,
                anchor_b,
                dir_a * accumulated[0],
                dir_b * accumulated[0],
            );
        }
    } else if (constraint.kind == CONSTRAINT_CONE) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = orthogonal_axis(i);
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, 0.0,
                mass_first, mass_second, first, second, accumulated[i], forces,
            );
            first = outcome.first;
            second = outcome.second;
            accumulated[i] = outcome.accumulated;
            forces = outcome.forces;
        }
        let angle = joint_coordinate(
            constraint,
            reference,
            first,
            second,
            anchor_a,
            anchor_b,
            0u,
        );
        if (angle > constraint.cone_angle) {
            let swing_axis = joint_axis(constraint, first, second, anchor_a, anchor_b, 0u);
            let outcome = limit_row(accumulated[3], swing_axis, mass_first, mass_second, first, second, forces);
            first = outcome.first;
            second = outcome.second;
            accumulated[3] = min(outcome.accumulated, 0.0);
            forces = outcome.forces;
        }
    } else if (constraint.kind == CONSTRAINT_SIXDOF) {
        let hinge = normalize(constraint.axis_a);
        let tangents = make_tangents(hinge);
        for (var i = 0u; i < 3u; i = i + 1u) {
            let world_axis = joint_dof_axis(constraint, first, i);
            let current = joint_coordinate(
                constraint,
                reference,
                first,
                second,
                anchor_a,
                anchor_b,
                i,
            );
            if (dof_locked(constraint.flags, i)) {
                let outcome = solve_point_row(world_axis, anchor_a, anchor_b, 0.0, mass_first, mass_second, first, second, accumulated[i], forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[i] = outcome.accumulated;
                forces = outcome.forces;
                continue;
            }
            if (dof_driven(constraint.flags, i)) {
                let drive = lane_drive(constraint, reference, first, second, anchor_a, anchor_b, i, params.dt);
                let outcome = solve_drive_row(drive, mass_first, mass_second, first, second, accumulated[i], forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[i] = outcome.accumulated;
                forces = outcome.forces;
            }
            if (dof_limited(constraint.flags, i)) {
                let min_goal = vec_index(constraint.linear_limit_min, i);
                let max_goal = vec_index(constraint.linear_limit_max, i);
                let row = dof_limit_row(i);
                if (current < min_goal) {
                    let outcome = solve_point_row(world_axis, anchor_a, anchor_b, min_goal, mass_first, mass_second, first, second, accumulated[row], forces);
                    first = outcome.first;
                    second = outcome.second;
                    accumulated[row] = min(outcome.accumulated, 0.0);
                    forces = outcome.forces;
                } else if (current > max_goal) {
                    let outcome = solve_point_row(world_axis, anchor_a, anchor_b, max_goal, mass_first, mass_second, first, second, accumulated[row], forces);
                    first = outcome.first;
                    second = outcome.second;
                    accumulated[row] = max(outcome.accumulated, 0.0);
                    forces = outcome.forces;
                }
            }
        }
        for (var i = 0u; i < 3u; i = i + 1u) {
            let world_axis = joint_dof_axis(constraint, first, 3u + i);
            let current = joint_coordinate(
                constraint,
                reference,
                first,
                second,
                anchor_a,
                anchor_b,
                3u + i,
            );
            let row = 3u + i;
            if (dof_locked(constraint.flags, row)) {
                let outcome = solve_angular_row(world_axis, mass_first, mass_second, first, second, accumulated[row], forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[row] = outcome.accumulated;
                forces = outcome.forces;
                continue;
            }
            if (dof_driven(constraint.flags, row)) {
                let drive = lane_drive(constraint, reference, first, second, anchor_a, anchor_b, row, params.dt);
                let outcome = solve_drive_row(drive, mass_first, mass_second, first, second, accumulated[row], forces);
                first = outcome.first;
                second = outcome.second;
                accumulated[row] = outcome.accumulated;
                forces = outcome.forces;
            }
            if (dof_limited(constraint.flags, row)) {
                let min_goal = vec_index(constraint.angular_limit_min, i);
                let max_goal = vec_index(constraint.angular_limit_max, i);
                let limit = dof_limit_row(row);
                if (current < min_goal) {
                    let outcome = limit_row(accumulated[limit], world_axis, mass_first, mass_second, first, second, forces);
                    first = outcome.first;
                    second = outcome.second;
                    accumulated[limit] = min(outcome.accumulated, 0.0);
                    forces = outcome.forces;
                } else if (current > max_goal) {
                    let outcome = limit_row(accumulated[limit], world_axis, mass_first, mass_second, first, second, forces);
                    first = outcome.first;
                    second = outcome.second;
                    accumulated[limit] = max(outcome.accumulated, 0.0);
                    forces = outcome.forces;
                }
            }
        }
    } else {
        for (var i = 0u; i < 3u; i = i + 1u) {
            let axis = orthogonal_axis(i);
            let outcome = solve_point_row(
                axis, anchor_a, anchor_b, 0.0,
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
            let outcome = solve_angular_row(basis[i], mass_first, mass_second, first, second, accumulated[3u + i], forces);
            first = outcome.first;
            second = outcome.second;
            accumulated[3u + i] = outcome.accumulated;
            forces = outcome.forces;
        }
    }
    if ((constraint.flags & CONSTRAINT_HAS_BREAK) != 0u ) {
        let force = length(forces.linear_second) / params.dt;
        let torque = length(forces.angular_second) / params.dt;
        if ((constraint.break_force > 0.0 && force > constraint.break_force)
            || (constraint.break_torque > 0.0 && torque > constraint.break_torque)) {
            runtime.broken = 1u;
        }
    }
    runtime.accumulated = accumulated;
    runtime.reaction = reaction_of(forces);
    constraint_runtime[constraint_index] = runtime;
    body_states[rows.first_row] = first.state;
    body_states[rows.second_row] = second.state;
    if (!residual) {
        return;
    }
    counter_max(COUNTER_SOLVE_LINEAR_RESIDUAL, solver_word(max(length(first.state.velocity - first_loaded.state.velocity), length(second.state.velocity - second_loaded.state.velocity)), SOLVER_VELOCITY_SCALE));
    counter_max(COUNTER_SOLVE_ANGULAR_RESIDUAL, solver_word(max(length(first.state.angular_velocity - first_loaded.state.angular_velocity), length(second.state.angular_velocity - second_loaded.state.angular_velocity)), SOLVER_VELOCITY_SCALE));
}

fn warm_point_impulse(axis: vec3f, point_a: vec3f, point_b: vec3f, impulse: f32, first: ptr<function, Body>, second: ptr<function, Body>) {
    apply_pair_impulse(first, second, point_a, point_b, axis * impulse);
}

fn warm_angular_impulse(axis: vec3f, impulse: f32, first: ptr<function, Body>, second: ptr<function, Body>) {
    let direction = normalize(axis);
    (*first).state.angular_velocity = (*first).state.angular_velocity - apply_inverse_inertia(*first, direction * impulse);
    (*second).state.angular_velocity = (*second).state.angular_velocity + apply_inverse_inertia(*second, direction * impulse);
}

fn warm_constraint_rows(
    constraint: ConstraintDescriptor,
    accumulated: array<f32, CONSTRAINT_ACCUMULATOR_SLOTS>,
    anchor_a: vec3f,
    anchor_b: vec3f,
    first: ptr<function, Body>,
    second: ptr<function, Body>,
) {
    let kind = constraint.kind;
    if (kind == CONSTRAINT_DISTANCE) {
        if ((constraint.flags & CONSTRAINT_IS_SPRING) == 0u) {
            warm_point_impulse(sign_normalize(anchor_b - anchor_a), anchor_a, anchor_b, accumulated[0], first, second);
        }
    } else if (kind == CONSTRAINT_REVOLUTE) {
        for (var index = 0u; index < 3u; index = index + 1u) {
            warm_point_impulse(orthogonal_axis(index), anchor_a, anchor_b, accumulated[index], first, second);
        }
        let local_hinge = normalize(constraint.axis_a);
        let tangents = constraint_local_frame(local_hinge);
        for (var index = 0u; index < 2u; index = index + 1u) {
            let local_axis = select(tangents.first, tangents.second, index == 1u);
            warm_angular_impulse(quat_rotate((*first).state.orientation, local_axis), accumulated[3u + index], first, second);
        }
        let hinge = quat_rotate((*first).state.orientation, local_hinge);
        if ((constraint.flags & CONSTRAINT_HAS_MOTOR) != 0u) {
            warm_angular_impulse(hinge, accumulated[6], first, second);
        }
        if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
            warm_angular_impulse(hinge, accumulated[5], first, second);
        }
    } else if (kind == CONSTRAINT_PRISMATIC) {
        let axis = normalize(quat_rotate((*first).state.orientation, constraint.axis_a));
        let tangents = make_tangents(axis);
        for (var index = 0u; index < 2u; index = index + 1u) {
            let tangent = select(tangents.first, tangents.second, index == 1u);
            warm_point_impulse(tangent, anchor_a, anchor_b, accumulated[index], first, second);
            warm_angular_impulse(tangent, accumulated[2u + index], first, second);
        }
        warm_angular_impulse(axis, accumulated[4], first, second);
        if ((constraint.flags & CONSTRAINT_HAS_MOTOR) != 0u) {
            warm_point_impulse(axis, anchor_a, anchor_b, accumulated[6], first, second);
        }
        if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
            warm_point_impulse(axis, anchor_a, anchor_b, accumulated[5], first, second);
        }
    } else if (kind == CONSTRAINT_BALL) {
        for (var index = 0u; index < 3u; index = index + 1u) {
            warm_point_impulse(orthogonal_axis(index), anchor_a, anchor_b, accumulated[index], first, second);
        }
        let hinge = quat_rotate((*first).state.orientation, normalize(constraint.axis_a));
        if ((constraint.flags & CONSTRAINT_HAS_LIMIT) != 0u) {
            warm_angular_impulse(hinge, accumulated[3], first, second);
        }
        if ((constraint.flags & CONSTRAINT_HAS_SWING) != 0u) {
            for (var index = 0u; index < 2u; index = index + 1u) {
                warm_angular_impulse(joint_axis(constraint, *first, *second, anchor_a, anchor_b, index), accumulated[4u + index], first, second);
            }
        }
    } else if (kind == CONSTRAINT_CONE) {
        for (var index = 0u; index < 3u; index = index + 1u) {
            warm_point_impulse(orthogonal_axis(index), anchor_a, anchor_b, accumulated[index], first, second);
        }
        warm_angular_impulse(joint_axis(constraint, *first, *second, anchor_a, anchor_b, 0u), accumulated[3], first, second);
    } else if (kind == CONSTRAINT_SIXDOF) {
        for (var index = 0u; index < 3u; index = index + 1u) {
            let world_axis = joint_dof_axis(constraint, *first, index);
            warm_point_impulse(world_axis, anchor_a, anchor_b, accumulated[index], first, second);
            warm_point_impulse(world_axis, anchor_a, anchor_b, accumulated[dof_limit_row(index)], first, second);
            let angular_axis = joint_dof_axis(constraint, *first, 3u + index);
            warm_angular_impulse(angular_axis, accumulated[3u + index], first, second);
            warm_angular_impulse(angular_axis, accumulated[dof_limit_row(3u + index)], first, second);
        }
    } else if (kind == CONSTRAINT_PULLEY) {
        let dir_a = sign_normalize(constraint.pulley_fixed_a - anchor_a);
        let dir_b = sign_normalize(constraint.pulley_fixed_b - anchor_b);
        let impulse = accumulated[0];
        (*first).state.velocity = (*first).state.velocity + dir_a * impulse * (*first).desc.inverse_mass;
        (*first).state.angular_velocity = (*first).state.angular_velocity + apply_inverse_inertia(*first, cross(anchor_a - body_com(*first), dir_a * impulse));
        (*second).state.velocity = (*second).state.velocity + dir_b * impulse * (*second).desc.inverse_mass;
        (*second).state.angular_velocity = (*second).state.angular_velocity + apply_inverse_inertia(*second, cross(anchor_b - body_com(*second), dir_b * impulse));
    } else if (kind == CONSTRAINT_GEAR) {
        let axis_a = normalize(quat_rotate((*first).state.orientation, constraint.axis_a));
        let axis_b = normalize(quat_rotate((*second).state.orientation, constraint.axis_b));
        let impulse = accumulated[0];
        (*first).state.angular_velocity = (*first).state.angular_velocity - apply_inverse_inertia(*first, axis_a * (constraint.gear_ratio * impulse));
        (*second).state.angular_velocity = (*second).state.angular_velocity + apply_inverse_inertia(*second, axis_b * impulse);
    } else {
        for (var index = 0u; index < 3u; index = index + 1u) {
            warm_point_impulse(orthogonal_axis(index), anchor_a, anchor_b, accumulated[index], first, second);
            warm_angular_impulse(orthogonal_axis(index), accumulated[3u + index], first, second);
        }
    }
}

fn warm_joint(constraint_index: u32) {
    let constraint = constraint_descs[constraint_index];
    let runtime = constraint_runtime[constraint_index];
    let rows = constraint_rows[constraint_index];
    if (runtime.broken != 0u) {
        return;
    }
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
    if ((constraint.flags & CONSTRAINT_WARM_START) != 0u) {
        warm_constraint_rows(
            constraint,
            runtime.accumulated,
            constraint_anchor(first, constraint.anchor_a),
            constraint_anchor(second, constraint.anchor_b),
            &first,
            &second,
        );
    }
    body_states[rows.first_row] = first.state;
    body_states[rows.second_row] = second.state;
}
