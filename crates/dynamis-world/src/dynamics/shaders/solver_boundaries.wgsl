@group(0) @binding(0) var<storage, read> segments: array<u32>;
@group(0) @binding(1) var<storage, read> a_bodies: array<u32>;
@group(0) @binding(2) var<storage, read> b_bodies: array<u32>;
@group(0) @binding(3) var<storage, read> a_payload: array<u32>;
@group(0) @binding(4) var<storage, read_write> first_a: array<u32>;
@group(0) @binding(5) var<storage, read_write> first_b: array<u32>;
@group(0) @binding(6) var<storage, read> contacts: array<Contact>;
@group(0) @binding(7) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(8) var<storage, read_write> block_counts: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> contact_counts: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
    let live = contact_blocks + segments[SOLVER_BLOCK_CONSTRAINT];
    let stride = grid_stride(groups);
    for (var slot = global_index(gid); slot < live; slot = slot + stride) {
        let first_body = a_bodies[slot];
        if (slot == 0u || a_bodies[slot - 1u] != first_body) {
            first_a[first_body] = slot + 1u;
        }
        let second_body = b_bodies[slot];
        if (slot == 0u || b_bodies[slot - 1u] != second_body) {
            first_b[second_body] = slot + 1u;
        }
        let block = a_payload[slot];
        var resolves = false;
        if (block < contact_blocks) {
            resolves = contact_block_resolves(contacts[block]);
        } else {
            resolves = constraint_runtime[block - contact_blocks].broken == 0u;
        }
        if (!resolves) {
            continue;
        }
        atomicAdd(&block_counts[first_body], 1u);
        atomicAdd(&block_counts[second_body], 1u);
        if (block < contact_blocks) {
            atomicAdd(&contact_counts[first_body], 1u);
            atomicAdd(&contact_counts[second_body], 1u);
        }
    }
}
