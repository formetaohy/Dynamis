@group(0) @binding(0) var<storage, read> contacts_raw: array<Contact>;
@group(0) @binding(1) var<storage, read> valid: array<u32>;
@group(0) @binding(2) var<storage, read> ranks: array<u32>;
@group(0) @binding(3) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(5) var<storage, read_write> a_body: array<u32>;
@group(0) @binding(6) var<storage, read> count_holder: array<u32>;

const BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= count_holder[0]) {
        return;
    }
    if (valid[index] == 0u) {
        return;
    }
    let dest = block_offsets[index / BLOCK_SIZE] + ranks[index];
    let contact = contacts_raw[index];
    contacts[dest] = contact;
    a_body[dest] = contact.a / 4u;
}
