@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read> contact_count: array<u32>;
@group(0) @binding(5) var<storage, read_write> contact_deltas: array<vec4f>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
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
        return;
    }
    let first = load_body(contact.a / MAX_COLLIDERS_PER_BODY);
    let second = load_body(contact.b / MAX_COLLIDERS_PER_BODY);
    let normal = contact.normal;
    var delta_a = vec3f(0.0);
    var delta_b = vec3f(0.0);
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
        delta_a = delta_a - normal * (correction * first.desc.inverse_mass);
        delta_b = delta_b + normal * (correction * second.desc.inverse_mass);
    }
    contact_deltas[index * 4u] = vec4f(delta_a, 0.0);
    contact_deltas[index * 4u + 1u] = vec4f(delta_b, 0.0);
}
