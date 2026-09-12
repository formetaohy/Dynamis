@group(0) @binding(0) var<storage, read> class_blocks: array<u32>;
@group(0) @binding(1) var<storage, read_write> class_bounds: array<u32>;
@group(0) @binding(2) var<storage, read> block_first_body: array<u32>;
@group(0) @binding(3) var<storage, read> block_second_body: array<u32>;
@group(0) @binding(4) var<storage, read_write> first_order_bodies: array<u32>;
@group(0) @binding(5) var<storage, read_write> first_order_blocks: array<u32>;
@group(0) @binding(6) var<storage, read> segments: array<u32>;
@group(0) @binding(11) var<storage, read> overflow_count: array<u32>;
@group(0) @binding(7) var<storage, read_write> block_counts: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> contact_counts: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> contacts: array<Contact>;
@group(0) @binding(10) var<storage, read> constraint_runtime: array<ConstraintRuntime>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let start = class_bounds[SOLVER_CLASS_OVERFLOW * SOLVER_CLASS_ROW_WORDS];
    let live = overflow_count[0];
    let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
    let stride = grid_stride(groups);
    for (var slot = global_index(gid); slot < live; slot = slot + stride) {
        let block = class_blocks[start + slot];
        let first_body = block_first_body[block];
        let second_body = block_second_body[block];
        first_order_bodies[slot] = first_body;
        first_order_blocks[slot] = block;
        var resolves = false;
        if (block < contact_blocks) {
            resolves = contact_block_resolves(contacts[block]);
        } else {
            resolves = constraint_runtime[block - contact_blocks].broken == 0u;
        }
        if (resolves) {
            atomicAdd(&block_counts[first_body], 1u);
            atomicAdd(&block_counts[second_body], 1u);
            if (block < contact_blocks) {
                atomicAdd(&contact_counts[first_body], 1u);
                atomicAdd(&contact_counts[second_body], 1u);
            }
        }
    }
}
