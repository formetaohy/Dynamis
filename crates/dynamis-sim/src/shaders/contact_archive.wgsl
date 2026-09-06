@group(0) @binding(0) var<storage, read> contact_indices: array<u32>;
@group(0) @binding(1) var<storage, read> contacts: array<Contact>;
@group(0) @binding(2) var<storage, read_write> prev_contacts: array<Contact>;
@group(0) @binding(3) var<storage, read_write> prev_contact_count: array<u32>;
@group(0) @binding(4) var<storage, read> contact_count: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= contact_count[0]) {
        return;
    }
    prev_contacts[index] = contacts[contact_indices[index]];
}
