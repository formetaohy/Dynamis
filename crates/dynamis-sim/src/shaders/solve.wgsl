@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(3) var<storage, read> contact_count: atomic<u32>;

fn solve_contact(index: u32) {
    let contact = contacts[index];
    if (contact.point_count == 0u) {
        return;
    }
    var first = bodies[contact.a];
    var second = bodies[contact.b];
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
    bodies[contact.a] = first;
    bodies[contact.b] = second;
}

@compute @workgroup_size(64u)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let count = atomicLoad(&contact_count);
    for (var index = lid.x; index < count; index = index + 64u) {
        solve_contact(index);
    }
}
