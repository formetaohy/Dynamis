@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> archive: array<Contact>;
@group(0) @binding(2) var<storage, read_write> contact_count: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = min(atomicLoad(&contact_count[0]), arrayLength(&contacts));
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {
        archive[index] = contacts[index];
    }
}
