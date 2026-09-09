@group(0) @binding(0) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> prev_contact_count: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> contacts: array<Contact>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x == 0u) {
        let archived = min(atomicLoad(&contact_count[0]), arrayLength(&contacts));
        atomicStore(&prev_contact_count[0], archived);
    }
}
