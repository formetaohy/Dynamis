@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> archive: array<Contact>;
@group(0) @binding(2) var<storage, read_write> contact_count: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= min(atomicLoad(&contact_count[0]), arrayLength(&contacts))) {
        return;
    }
    archive[index] = contacts[index];
}
