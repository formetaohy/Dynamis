@group(0) @binding(0) var<storage, read> contact_count: array<u32>;
@group(0) @binding(1) var<storage, read_write> prev_contact_count: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x == 0u) {
        prev_contact_count[0] = contact_count[0];
    }
}
