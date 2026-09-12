@group(0) @binding(0) var<storage, read_write> class_tokens: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = arrayLength(&class_tokens);
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {
        atomicStore(&class_tokens[index], 0u);
    }
}
