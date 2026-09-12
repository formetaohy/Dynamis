@group(0) @binding(0) var<storage, read> a_payload: array<u32>;
@group(0) @binding(1) var<storage, read> block_second_body: array<u32>;
@group(0) @binding(2) var<storage, read_write> b_bodies: array<u32>;
@group(0) @binding(3) var<storage, read_write> b_blocks: array<u32>;
@group(0) @binding(4) var<storage, read> block_count: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = block_count[0];
    let stride = grid_stride(groups);
    for (var slot = global_index(gid); slot < live; slot = slot + stride) {
        b_bodies[slot] = block_second_body[a_payload[slot]];
        b_blocks[slot] = slot;
    }
}
