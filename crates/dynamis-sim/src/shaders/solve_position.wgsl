@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(3) var<storage, read> contact_count: atomic<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= atomicLoad(&contact_count)) {
        return;
    }
    let contact = contacts[index];
    if (contact.point_count == 0u || contact.sensor == 1u) {
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
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        let depth = contact.points[point_index].depth;
        let position = contact.points[point_index].position;
        let error = max(depth - params.slop, 0.0);
        if (error <= 0.0) {
            continue;
        }
        let k = point_momentum_mass(first, second, position, position, normal);
        if (k == 0.0) {
            continue;
        }
        let correction = params.relaxation * error / k / f32(max(params.position_iterations, 1u));
        first.position = first.position - normal * (correction * first.inverse_mass);
        second.position = second.position + normal * (correction * second.inverse_mass);
    }
    first.inverse_mass = first_original.inverse_mass;
    first.inverse_inertia_body = first_original.inverse_inertia_body;
    second.inverse_mass = second_original.inverse_mass;
    second.inverse_inertia_body = second_original.inverse_inertia_body;
    bodies[first_slot] = first;
    bodies[second_slot] = second;
}
