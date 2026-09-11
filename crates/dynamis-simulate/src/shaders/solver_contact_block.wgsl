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
    var updated = contact;
    let normal = contact.normal;
    let tangents = make_tangents(normal);
    var normal_accum = 0.0;
    var contact_updated = false;
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        contact_updated = true;
        normal_accum = normal_accum + contact.points[point_index].accumulated_normal;
        let position = contact.points[point_index].position;
        let velocity = relative_velocity(first, second, position, position);
        let normal_speed = dot(velocity, normal);
        if (normal_speed < 0.0) {
            let normal_mass = point_momentum_mass(pair.split_first, pair.split_second, position, position, normal);
            var accumulated = contact.points[point_index].accumulated_normal;
            var restitution = contact.restitution;
            if (normal_speed > -params.restitution_threshold) {
                restitution = 0.0;
            }
            let goal_speed = -normal_speed * (1.0 + restitution);
            let delta = goal_speed / normal_mass;
            let next = max(0.0, accumulated + delta);
            let applied = next - accumulated;
            accumulated = next;
            apply_pair_impulse(&first, &second, position, position, normal * applied);
            let velocity_after = relative_velocity(first, second, position, position);
            let tangent_speed = dot(velocity_after, tangents.first);
            let tangent_mass =
                point_momentum_mass(pair.split_first, pair.split_second, position, position, tangents.first);
            let friction = contact.friction;
            let friction_limit = friction * accumulated;
            var accumulated_tangent_1 = contact.points[point_index].accumulated_tangent_1;
            var next_tangent_1 =
                clamp(accumulated_tangent_1 - tangent_speed / tangent_mass, -friction_limit, friction_limit);
            let applied_tangent_1 = next_tangent_1 - accumulated_tangent_1;
            accumulated_tangent_1 = next_tangent_1;
            apply_pair_impulse(&first, &second, position, position, tangents.first * applied_tangent_1);
            let velocity_after_2 = relative_velocity(first, second, position, position);
            let tangent_speed_2 = dot(velocity_after_2, tangents.second);
            let tangent_mass_2 =
                point_momentum_mass(pair.split_first, pair.split_second, position, position, tangents.second);
            var accumulated_tangent_2 = contact.points[point_index].accumulated_tangent_2;
            let remaining = sqrt(max(friction_limit * friction_limit - accumulated_tangent_1 * accumulated_tangent_1, 0.0));
            var next_tangent_2 =
                clamp(accumulated_tangent_2 - tangent_speed_2 / tangent_mass_2, -remaining, remaining);
            let applied_tangent_2 = next_tangent_2 - accumulated_tangent_2;
            accumulated_tangent_2 = next_tangent_2;
            apply_pair_impulse(&first, &second, position, position, tangents.second * applied_tangent_2);
            updated.points[point_index].accumulated_normal = accumulated;
            updated.points[point_index].accumulated_tangent_1 = accumulated_tangent_1;
            updated.points[point_index].accumulated_tangent_2 = accumulated_tangent_2;
        }
    }
    if (contact_updated) {
        let rel_spin = second.state.angular_velocity - first.state.angular_velocity;
        let planar_spin = rel_spin - normal * dot(rel_spin, normal);
        let planar_len = length(planar_spin);
        if (planar_len > 1e-6) {
            let axis = planar_spin / planar_len;
            let mass = dot(axis, apply_inverse_inertia(pair.split_first, axis))
                + dot(axis, apply_inverse_inertia(pair.split_second, axis));
            if (mass > 0.0) {
                let limit = contact.rolling_friction * normal_accum;
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
                let limit = contact.spin_friction * normal_accum;
                let applied = clamp(-twist / mass, -limit, limit);
                first.state.angular_velocity =
                    first.state.angular_velocity - apply_inverse_inertia(first, normal * applied);
                second.state.angular_velocity =
                    second.state.angular_velocity + apply_inverse_inertia(second, normal * applied);
            }
        }
        contacts[contact_index] = updated;
    }
    store_block_delta(
        slot,
        first.state.velocity - pair.first.state.velocity,
        first.state.angular_velocity - pair.first.state.angular_velocity,
        second.state.velocity - pair.second.state.velocity,
        second.state.angular_velocity - pair.second.state.angular_velocity,
    );
    wake_on_impact(first_slot, pair.first, pair.second);
    wake_on_impact(second_slot, pair.second, pair.first);
}
