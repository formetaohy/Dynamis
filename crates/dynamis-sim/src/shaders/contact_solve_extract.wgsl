@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(3) var<storage, read> contact_count: array<u32>;
@group(0) @binding(4) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> contact_deltas: array<vec4f>;

fn wake_on_impact(
    slot: u32,
    sleeping: RigidBody,
    moving: RigidBody,
) {
    if ((sleeping.flags & BODY_SLEEPING) == 0u || (moving.flags & BODY_SLEEPING) != 0u) {
        return;
    }
    let velocity = moving.velocity - sleeping.velocity;
    let spin = moving.angular_velocity - sleeping.angular_velocity;
    if (length(velocity) > params.wake_velocity || length(spin) > params.wake_velocity) {
        atomicOr(&wake_flags[slot], 1u);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= contact_count[0]) {
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
    let first_slot = contact.a / 4u;
    let second_slot = contact.b / 4u;
    let first_original = bodies[first_slot];
    let second_original = bodies[second_slot];
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
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        let position = contact.points[point_index].position;
        let velocity = relative_velocity(first, second, position, position);
        let normal_speed = dot(velocity, normal);
        if (normal_speed < 0.0) {
            let normal_mass = point_momentum_mass(first, second, position, position, normal);
            var accumulated = contact.points[point_index].accumulated_normal;
            var restitution = max(first.restitution, second.restitution);
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
            let friction = sqrt(first.friction * second.friction);
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
    let delta_a = first.velocity - first_original.velocity;
    let delta_spin_a = first.angular_velocity - first_original.angular_velocity;
    let delta_b = second.velocity - second_original.velocity;
    let delta_spin_b = second.angular_velocity - second_original.angular_velocity;
    contact_deltas[index * 4u] = vec4f(delta_a, 0.0);
    contact_deltas[index * 4u + 1u] = vec4f(delta_spin_a, 0.0);
    contact_deltas[index * 4u + 2u] = vec4f(delta_b, 0.0);
    contact_deltas[index * 4u + 3u] = vec4f(delta_spin_b, 0.0);
    wake_on_impact(
        first_slot,
        first_original,
        second_original,
    );
    wake_on_impact(
        second_slot,
        second_original,
        first_original,
    );
}
