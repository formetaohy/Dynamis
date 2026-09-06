@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read> contact_count: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> bodies: array<RigidBody>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= atomicLoad(&contact_count)) {
        return;
    }
    let contact = contacts[index];
    if (contact.point_count == 0u) {
        return;
    }
    let normal = contact.normal;
    let tangents = make_tangents(normal);
    var first = bodies[contact.a];
    var second = bodies[contact.b];
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        let point = contact.points[point_index];
        let position = point.position;
        apply_pair_impulse(
            &first,
            &second,
            position,
            position,
            normal * point.accumulated_normal,
        );
        apply_pair_impulse(
            &first,
            &second,
            position,
            position,
            tangents.first * point.accumulated_tangent_1 + tangents.second * point.accumulated_tangent_2,
        );
    }
    bodies[contact.a] = first;
    bodies[contact.b] = second;
}
