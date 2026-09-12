@group(0) @binding(0) var<storage, read> first_order_blocks: array<u32>;
@group(0) @binding(1) var<storage, read> block_second_body: array<u32>;
@group(0) @binding(4) var<storage, read> overflow_count: array<u32>;
@group(0) @binding(2) var<storage, read_write> second_order_bodies: array<u32>;
@group(0) @binding(3) var<storage, read_write> second_order_blocks: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = overflow_count[0];
    let stride = grid_stride(groups);
    for (var slot = global_index(gid); slot < live; slot = slot + stride) {
        second_order_bodies[slot] = block_second_body[first_order_blocks[slot]];
        second_order_blocks[slot] = slot;
    }
}
