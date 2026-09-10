@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> deltas: array<vec4f>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= min(atomicLoad(&contact_count[0]), arrayLength(&contacts))) {
        return;
    }
    let contact = contacts[index];
    if (contact.point_count == 0u || contact.sensor == 1u) {
        deltas[index * 4u] = vec4f(0.0);
        deltas[index * 4u + 1u] = vec4f(0.0);
        return;
    }
    let first_loaded = load_body(contact.a / MAX_COLLIDERS_PER_BODY);
    let second_loaded = load_body(contact.b / MAX_COLLIDERS_PER_BODY);
    var first = first_loaded;
    var second = second_loaded;
    if (body_is_inert(first_loaded)) {
        first = body_frozen(first_loaded);
    }
    if (body_is_inert(second_loaded)) {
        second = body_frozen(second_loaded);
    }
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
    deltas[index * 4u] = vec4f(delta_a, 0.0);
    deltas[index * 4u + 1u] = vec4f(delta_b, 0.0);
}
