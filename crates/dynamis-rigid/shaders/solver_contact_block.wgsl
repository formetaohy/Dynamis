

fn tangential_mass(mass: f32, contact: Contact) -> f32 {
    return mass * f32(max(contact.point_count, 1u));
}

fn solve_contact_block(contact_index: u32, slot: u32) {
    let contact = contacts[contact_index];
    let first_slot = blocks[slot * 2u];
    let second_slot = blocks[slot * 2u + 1u];
    if (!contact_block_resolves(contact)) {
        commit_block(slot, first_slot, second_slot, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
        return;
    }
    let pair = block_bodies(first_slot, second_slot);
    if (body_is_inert(pair.first) && body_is_inert(pair.second)) {
        commit_block(slot, first_slot, second_slot, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
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
    var impulses: array<vec4f, CONTACT_MAX_POINTS>;
    for (var point_index = 0u; point_index < CONTACT_MAX_POINTS; point_index = point_index + 1u) {
        impulses[point_index] = vec4f(0.0);
    }
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        let point = contact.points[point_index];
        let anchor_first = manifold_arm(contact, pair.first, point, true);
        let anchor_second = manifold_arm(contact, pair.second, point, false);
        var accumulated_normal = point.accumulated_normal;
        var accumulated_tangent_1 = point.accumulated_tangent_1;
        var accumulated_tangent_2 = point.accumulated_tangent_2;
        let normal_speed =
            dot(relative_velocity(pair.first, pair.second, anchor_first, anchor_second), normal);
        let normal_mass =
            point_momentum_mass(pair.split_first, pair.split_second, anchor_first, anchor_second, normal);
        let delta = (target_speeds[contact_index * CONTACT_MAX_POINTS + point_index] - normal_speed) / normal_mass;
        let next_normal = max(0.0, accumulated_normal + delta);
        impulses[point_index] = impulses[point_index] + vec4f(normal * (next_normal - accumulated_normal), 0.0);
        accumulated_normal = next_normal;
        let friction_limit = contact.friction * accumulated_normal;
        let tangent_1_mass = tangential_mass(
            point_momentum_mass(
                pair.split_first,
                pair.split_second,
                anchor_first,
                anchor_second,
                tangents.first,
            ),
            contact,
        );
        let tangent_1_speed = dot(
            relative_velocity(pair.first, pair.second, anchor_first, anchor_second),
            tangents.first,
        );
        let next_tangent_1 =
            clamp(accumulated_tangent_1 - tangent_1_speed / tangent_1_mass, -friction_limit, friction_limit);
        impulses[point_index] =
            impulses[point_index] + vec4f(tangents.first * (next_tangent_1 - accumulated_tangent_1), 0.0);
        accumulated_tangent_1 = next_tangent_1;
        let tangent_2_mass = tangential_mass(
            point_momentum_mass(
                pair.split_first,
                pair.split_second,
                anchor_first,
                anchor_second,
                tangents.second,
            ),
            contact,
        );
        let tangent_2_speed = dot(
            relative_velocity(pair.first, pair.second, anchor_first, anchor_second),
            tangents.second,
        );
        let remaining = sqrt(
            max(friction_limit * friction_limit - accumulated_tangent_1 * accumulated_tangent_1, 0.0),
        );
        let next_tangent_2 =
            clamp(accumulated_tangent_2 - tangent_2_speed / tangent_2_mass, -remaining, remaining);
        impulses[point_index] =
            impulses[point_index] + vec4f(tangents.second * (next_tangent_2 - accumulated_tangent_2), 0.0);
        accumulated_tangent_2 = next_tangent_2;
        updated.points[point_index].accumulated_normal = accumulated_normal;
        updated.points[point_index].accumulated_tangent_1 = accumulated_tangent_1;
        updated.points[point_index].accumulated_tangent_2 = accumulated_tangent_2;
    }
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        let point = contact.points[point_index];
        let anchor_first = manifold_arm(contact, pair.first, point, true);
        let anchor_second = manifold_arm(contact, pair.second, point, false);
        apply_pair_impulse(&first, &second, anchor_first, anchor_second, impulses[point_index].xyz);
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
    }
    commit_block(
        slot,
        first_slot,
        second_slot,
        first.state.velocity - pair.first.state.velocity,
        first.state.angular_velocity - pair.first.state.angular_velocity,
        second.state.velocity - pair.second.state.velocity,
        second.state.angular_velocity - pair.second.state.angular_velocity,
    );
}

fn warm_contact_block(contact_index: u32, slot: u32) {
    let contact = contacts[contact_index];
    let first_slot = blocks[slot * 2u];
    let second_slot = blocks[slot * 2u + 1u];
    if (!contact_block_resolves(contact)) {
        commit_block(slot, first_slot, second_slot, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
        return;
    }
    let first_loaded = load_body(first_slot);
    let second_loaded = load_body(second_slot);
    var first = first_loaded;
    var second = second_loaded;
    if (body_is_inert(first_loaded)) {
        first = body_frozen(first_loaded);
    }
    if (body_is_inert(second_loaded)) {
        second = body_frozen(second_loaded);
    }
    let tangents = make_tangents(contact.normal);
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        let point = contact.points[point_index];
        let anchor_first = manifold_arm(contact, first, point, true);
        let anchor_second = manifold_arm(contact, second, point, false);
        let impulse = contact.normal * point.accumulated_normal
            + tangents.first * point.accumulated_tangent_1
            + tangents.second * point.accumulated_tangent_2;
        apply_pair_impulse(&first, &second, anchor_first, anchor_second, impulse);
    }
    commit_block(
        slot,
        first_slot,
        second_slot,
        first.state.velocity - first_loaded.state.velocity,
        first.state.angular_velocity - first_loaded.state.angular_velocity,
        second.state.velocity - second_loaded.state.velocity,
        second.state.angular_velocity - second_loaded.state.angular_velocity,
    );
}

