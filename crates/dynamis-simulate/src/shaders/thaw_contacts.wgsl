@group(0) @binding(0) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(1) var<storage, read> resting: array<Contact>;
@group(0) @binding(2) var<storage, read_write> resting_live: array<u32>;
@group(0) @binding(3) var<storage, read_write> resting_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> resting_next: array<u32>;
@group(0) @binding(5) var<storage, read_write> resting_free: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> body_descs: array<BodyDescriptor>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= min(atomicLoad(&resting_count[0]), arrayLength(&resting))) {
        return;
    }
    if (resting_live[index] == 0u) {
        return;
    }
    let contact = resting[index];
    let first = contact.a / MAX_COLLIDERS_PER_BODY;
    let second = contact.b / MAX_COLLIDERS_PER_BODY;
    if (!body_is_active(body_states[first], body_descs[first]) &&
        !body_is_active(body_states[second], body_descs[second])) {
        return;
    }
    resting_live[index] = 0u;
    resting_next[index] = atomicExchange(&resting_free[0], index);
}
