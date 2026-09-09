@group(0) @binding(0) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(1) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(2) var<storage, read> contacts: array<Contact>;
@group(0) @binding(3) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> island_parents: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> wake_flags: array<atomic<u32>>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn island_link(a: u32, b: u32) {
    atomicMin(&island_parents[a], b);
    atomicMin(&island_parents[b], a);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= min(atomicLoad(&contact_count[0]), arrayLength(&contacts))) {
        return;
    }
    let contact = contacts[index];
    if (contact.point_count == 0u || contact.sensor == 1u) {
        return;
    }
    let first_slot = contact.a / MAX_COLLIDERS_PER_BODY;
    let second_slot = contact.b / MAX_COLLIDERS_PER_BODY;
    let first = load_body(first_slot);
    let second = load_body(second_slot);
    let first_static = body_is_static(first);
    let second_static = body_is_static(second);
    if (first_static && atomicLoad(&wake_flags[first_slot]) != 0u) {
        atomicOr(&wake_flags[second_slot], 1u);
    }
    if (second_static && atomicLoad(&wake_flags[second_slot]) != 0u) {
        atomicOr(&wake_flags[first_slot], 1u);
    }
    if (!first_static && !second_static &&
        body_is_dynamic(first) && first.state.sleeping == 0u &&
        body_is_dynamic(second) && second.state.sleeping == 0u) {
        island_link(first_slot, second_slot);
    }
}
