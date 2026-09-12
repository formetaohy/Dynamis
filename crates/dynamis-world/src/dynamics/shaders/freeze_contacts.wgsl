@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(3) var<storage, read_write> resting: array<Contact>;
@group(0) @binding(4) var<storage, read_write> resting_live: array<u32>;
@group(0) @binding(5) var<storage, read_write> resting_count: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> spillover: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(8) var<storage, read_write> resting_next: array<u32>;
@group(0) @binding(9) var<storage, read_write> resting_free: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read> collider_owners: array<u32>;

fn acquire_slot() -> u32 {
    var slot = 0u;
    loop {
        let head = atomicLoad(&resting_free[0]);
        if (head == NO_SLOT) {
            slot = atomicAdd(&resting_count[0], 1u);
            break;
        }
        let next = resting_next[head];
        if (atomicCompareExchangeWeak(&resting_free[0], head, next).exchanged) {
            slot = head;
            break;
        }
    }
    return slot;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = min(atomicLoad(&contact_count[0]), arrayLength(&contacts));
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {
        let contact = contacts[index];
        let first = collider_owners[contact.a];
        let second = collider_owners[contact.b];
        if (body_is_active(body_states[first], body_descs[first]) ||
            body_is_active(body_states[second], body_descs[second])) {
            continue;
        }
        let slot = acquire_slot();
        if (slot >= arrayLength(&resting)) {
            atomicAdd(&spillover[0], 1u);
            continue;
        }
        resting[slot] = contact;
        resting_live[slot] = 1u;
    }
}
