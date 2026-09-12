@group(0) @binding(0) var<storage, read_write> class_tokens: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> class_cursors: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> class_blocks: array<u32>;
@group(0) @binding(3) var<storage, read> segments: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = block_live();
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {
        let token = atomicLoad(&class_tokens[index]);
        if (token == 0u) {
            continue;
        }
        let slot = atomicAdd(&class_cursors[token_class(token)], 1u);
        class_blocks[slot] = index;
    }
}
