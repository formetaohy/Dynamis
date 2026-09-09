@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> contact_deltas: array<vec4f>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

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

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= min(atomicLoad(&contact_count[0]), arrayLength(&contacts))) {
        return;
    }
    let contact = contacts[index];
    if (contact.point_count == 0u || contact.sensor == 1u) {
        contact_deltas[index * 4u] = vec4f(0.0);
        contact_deltas[index * 4u + 1u] = vec4f(0.0);
        contact_deltas[index * 4u + 2u] = vec4f(0.0);
        contact_deltas[index * 4u + 3u] = vec4f(0.0);
        return;
    }
    let first_slot = contact.a / MAX_COLLIDERS_PER_BODY;
    let second_slot = contact.b / MAX_COLLIDERS_PER_BODY;
    let first_original = load_body(first_slot);
    let second_original = load_body(second_slot);
    var first = first_original;
    var second = second_original;
    if (body_is_inert(first_original)) {
        first = body_frozen(first_original);
    }
    if (body_is_inert(second_original)) {
        second = body_frozen(second_original);
    }
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
            let normal_mass = point_momentum_mass(first, second, position, position, normal);
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
            let tangent_mass = point_momentum_mass(first, second, position, position, tangents.first);
            let friction = contact.friction;
            let friction_limit = friction * accumulated;
            var accumulated_tangent_1 = contact.points[point_index].accumulated_tangent_1;
            var next_tangent_1 = clamp(accumulated_tangent_1 - tangent_speed / tangent_mass, -friction_limit, friction_limit);
            let applied_tangent_1 = next_tangent_1 - accumulated_tangent_1;
            accumulated_tangent_1 = next_tangent_1;
            apply_pair_impulse(&first, &second, position, position, tangents.first * applied_tangent_1);
            let velocity_after_2 = relative_velocity(first, second, position, position);
            let tangent_speed_2 = dot(velocity_after_2, tangents.second);
            let tangent_mass_2 = point_momentum_mass(first, second, position, position, tangents.second);
            var accumulated_tangent_2 = contact.points[point_index].accumulated_tangent_2;
            let remaining = sqrt(max(friction_limit * friction_limit - accumulated_tangent_1 * accumulated_tangent_1, 0.0));
            var next_tangent_2 = clamp(accumulated_tangent_2 - tangent_speed_2 / tangent_mass_2, -remaining, remaining);
            let applied_tangent_2 = next_tangent_2 - accumulated_tangent_2;
            accumulated_tangent_2 = next_tangent_2;
            apply_pair_impulse(&first, &second, position, position, tangents.second * applied_tangent_2);
            var updated_contact = contacts[index];
            updated_contact.points[point_index].accumulated_normal = accumulated;
            updated_contact.points[point_index].accumulated_tangent_1 = accumulated_tangent_1;
            updated_contact.points[point_index].accumulated_tangent_2 = accumulated_tangent_2;
            contacts[index] = updated_contact;
        }
    }
    if (contact_updated) {
        let rel_spin = second.state.angular_velocity - first.state.angular_velocity;
        let planar_spin = rel_spin - normal * dot(rel_spin, normal);
        let planar_len = length(planar_spin);
        if (planar_len > 1e-6) {
            let axis = planar_spin / planar_len;
            let k = dot(axis, apply_inverse_inertia(first, axis)) + dot(axis, apply_inverse_inertia(second, axis));
            if (k > 0.0) {
                let limit = contact.rolling_friction * normal_accum;
                let applied = clamp(-planar_len / k, -limit, limit);
                first.state.angular_velocity = first.state.angular_velocity - apply_inverse_inertia(first, axis * applied);
                second.state.angular_velocity = second.state.angular_velocity + apply_inverse_inertia(second, axis * applied);
            }
        }
        let twist = dot(rel_spin, normal);
        if (abs(twist) > 1e-6) {
            let k = dot(normal, apply_inverse_inertia(first, normal)) + dot(normal, apply_inverse_inertia(second, normal));
            if (k > 0.0) {
                let limit = contact.spin_friction * normal_accum;
                let applied = clamp(-twist / k, -limit, limit);
                first.state.angular_velocity = first.state.angular_velocity - apply_inverse_inertia(first, normal * applied);
                second.state.angular_velocity = second.state.angular_velocity + apply_inverse_inertia(second, normal * applied);
            }
        }
    }
    let delta_a = first.state.velocity - first_original.state.velocity;
    let delta_spin_a = first.state.angular_velocity - first_original.state.angular_velocity;
    let delta_b = second.state.velocity - second_original.state.velocity;
    let delta_spin_b = second.state.angular_velocity - second_original.state.angular_velocity;
    contact_deltas[index * 4u] = vec4f(delta_a, 0.0);
    contact_deltas[index * 4u + 1u] = vec4f(delta_spin_a, 0.0);
    contact_deltas[index * 4u + 2u] = vec4f(delta_b, 0.0);
    contact_deltas[index * 4u + 3u] = vec4f(delta_spin_b, 0.0);
    wake_on_impact(first_slot, first_original, second_original);
    wake_on_impact(second_slot, second_original, first_original);
}
