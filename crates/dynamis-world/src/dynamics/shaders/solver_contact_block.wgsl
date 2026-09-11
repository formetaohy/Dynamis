
fn wake_on_impact(slot: u32, sleeping: Body, moving: Body) {
    if (sleeping.state.sleeping == 0u || moving.state.sleeping != 0u) {
        return;
    }
    let velocity = moving.state.velocity - sleeping.state.velocity;
    let spin = moving.state.angular_velocity - sleeping.state.angular_velocity;
    if (length(velocity) > params.wake_velocity || length(spin) > params.wake_velocity) {
        atomicOr(&wake_flags[slot], 1u);
    }
}

fn solve_contact_block(contact_index: u32, slot: u32) {
    let contact = contacts[contact_index];
    if (!contact_block_resolves(contact)) {
        store_block_delta(slot, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
        return;
    }
    let first_slot = contact.a / MAX_COLLIDERS_PER_BODY;
    let second_slot = contact.b / MAX_COLLIDERS_PER_BODY;
    let pair = block_pair(first_slot, second_slot);
    if (body_is_inert(pair.first) && body_is_inert(pair.second)) {
        store_block_delta(slot, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
        return;
    }
    var first = pair.first;
    var second = pair.second;
    let normal = contact.normal;
    let tangents = make_tangents(normal);
    var updated = contact;
    var normal_total = 0.0;
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        normal_total = normal_total + max(contact.points[point_index].accumulated_normal, 0.0);
    }
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        let point = contact.points[point_index];
        let position = point.position;
        var accumulated_normal = point.accumulated_normal;
        var accumulated_tangent_1 = point.accumulated_tangent_1;
        var accumulated_tangent_2 = point.accumulated_tangent_2;
        let velocity = relative_velocity(first, second, position, position);
        let normal_speed = dot(velocity, normal);
        let normal_mass = point_momentum_mass(pair.split_first, pair.split_second, position, position, normal);
        let delta = (point.target_speed - normal_speed) / normal_mass;
        let next_normal = max(0.0, accumulated_normal + delta);
        apply_pair_impulse(&first, &second, position, position, normal * (next_normal - accumulated_normal));
        accumulated_normal = next_normal;
        let friction_limit = contact.friction * accumulated_normal;
        let tangent_1_mass =
            point_momentum_mass(pair.split_first, pair.split_second, position, position, tangents.first);
        let tangent_1_speed = dot(relative_velocity(first, second, position, position), tangents.first);
        let next_tangent_1 =
            clamp(accumulated_tangent_1 - tangent_1_speed / tangent_1_mass, -friction_limit, friction_limit);
        apply_pair_impulse(&first, &second, position, position, tangents.first * (next_tangent_1 - accumulated_tangent_1));
        accumulated_tangent_1 = next_tangent_1;
        let tangent_2_mass =
            point_momentum_mass(pair.split_first, pair.split_second, position, position, tangents.second);
        let tangent_2_speed = dot(relative_velocity(first, second, position, position), tangents.second);
        let remaining = sqrt(max(friction_limit * friction_limit - accumulated_tangent_1 * accumulated_tangent_1, 0.0));
        let next_tangent_2 =
            clamp(accumulated_tangent_2 - tangent_2_speed / tangent_2_mass, -remaining, remaining);
        apply_pair_impulse(&first, &second, position, position, tangents.second * (next_tangent_2 - accumulated_tangent_2));
        accumulated_tangent_2 = next_tangent_2;
        updated.points[point_index].accumulated_normal = accumulated_normal;
        updated.points[point_index].accumulated_tangent_1 = accumulated_tangent_1;
        updated.points[point_index].accumulated_tangent_2 = accumulated_tangent_2;
    }
    {
        let rel_spin = second.state.angular_velocity - first.state.angular_velocity;
        let planar_spin = rel_spin - normal * dot(rel_spin, normal);
        let planar_len = length(planar_spin);
        if (planar_len > 1e-6) {
            let axis = planar_spin / planar_len;
            let mass = dot(axis, apply_inverse_inertia(pair.split_first, axis))
                + dot(axis, apply_inverse_inertia(pair.split_second, axis));
            if (mass > 0.0) {
                let limit = contact.rolling_friction * normal_total;
                let applied = clamp(-planar_len / mass, -limit, limit);
                first.state.angular_velocity =
                    first.state.angular_velocity - apply_inverse_inertia(first, axis * applied);
                second.state.angular_velocity =
                    second.state.angular_velocity + apply_inverse_inertia(second, axis * applied);
            }
        }
        let twist = dot(rel_spin, normal);
        if (abs(twist) > 1e-6) {
            let mass = dot(normal, apply_inverse_inertia(pair.split_first, normal))
                + dot(normal, apply_inverse_inertia(pair.split_second, normal));
            if (mass > 0.0) {
                let limit = contact.spin_friction * normal_total;
                let applied = clamp(-twist / mass, -limit, limit);
                first.state.angular_velocity =
                    first.state.angular_velocity - apply_inverse_inertia(first, normal * applied);
                second.state.angular_velocity =
                    second.state.angular_velocity + apply_inverse_inertia(second, normal * applied);
            }
        }
        contacts[contact_index] = updated;
        wake_on_impact(first_slot, pair.first, pair.second);
        wake_on_impact(second_slot, pair.second, pair.first);
    }
    store_block_delta(
        slot,
        first.state.velocity - pair.first.state.velocity,
        first.state.angular_velocity - pair.first.state.angular_velocity,
        second.state.velocity - pair.second.state.velocity,
        second.state.angular_velocity - pair.second.state.angular_velocity,
    );
}
