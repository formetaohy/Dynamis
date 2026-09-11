@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read> segments: array<u32>;
@group(0) @binding(5) var<storage, read> a_payload: array<u32>;
@group(0) @binding(6) var<storage, read_write> block_deltas: array<vec4f>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn store_block_delta(slot: u32, delta_a: vec3f, spin_a: vec3f, delta_b: vec3f, spin_b: vec3f) {
    block_deltas[slot * 4u] = vec4f(delta_a, 0.0);
    block_deltas[slot * 4u + 1u] = vec4f(spin_a, 0.0);
    block_deltas[slot * 4u + 2u] = vec4f(delta_b, 0.0);
    block_deltas[slot * 4u + 3u] = vec4f(spin_b, 0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let slot = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
    if (slot >= contact_blocks) {
        store_block_delta(slot, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
        return;
    }
    let contact = contacts[a_payload[slot]];
    if (!contact_block_resolves(contact)) {
        store_block_delta(slot, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
        return;
    }
    let first_slot = contact.a / MAX_COLLIDERS_PER_BODY;
    let second_slot = contact.b / MAX_COLLIDERS_PER_BODY;
    var first = load_body(first_slot);
    var second = load_body(second_slot);
    if (body_is_inert(first)) {
        first = body_frozen(first);
    }
    if (body_is_inert(second)) {
        second = body_frozen(second);
    }
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        let point = contact.points[point_index];
        apply_pair_impulse(
            &first,
            &second,
            point.position,
            point.position,
            contact.normal * point.accumulated_normal,
        );
    }
    store_block_delta(
        slot,
        first.state.velocity - body_states[first_slot].velocity,
        first.state.angular_velocity - body_states[first_slot].angular_velocity,
        second.state.velocity - body_states[second_slot].velocity,
        second.state.angular_velocity - body_states[second_slot].angular_velocity,
    );
}
