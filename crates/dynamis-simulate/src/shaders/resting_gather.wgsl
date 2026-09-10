@group(0) @binding(0) var<storage, read> resting: array<Contact>;
@group(0) @binding(1) var<storage, read_write> resting_count: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> resting_live: array<u32>;
@group(0) @binding(3) var<storage, read_write> index_major: array<u32>;
@group(0) @binding(4) var<storage, read_write> index_minor: array<u32>;
@group(0) @binding(5) var<storage, read_write> index_slots: array<u32>;
@group(0) @binding(6) var<storage, read_write> gathered: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= min(atomicLoad(&resting_count[0]), arrayLength(&resting_live))) {
        return;
    }
    if (resting_live[index] == 0u) {
        return;
    }
    let contact = resting[index];
    let slot = atomicAdd(&gathered[0], 1u);
    if (slot >= arrayLength(&index_slots)) {
        return;
    }
    index_major[slot] = contact.first_body_id;
    index_minor[slot] = contact.second_body_id;
    index_slots[slot] = index;
}
