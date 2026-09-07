@group(0) @binding(0) var<storage, read> contact_a_body: array<u32>;
@group(0) @binding(1) var<storage, read> contact_keys_b: array<u32>;
@group(0) @binding(2) var<storage, read> contact_count: array<u32>;
@group(0) @binding(3) var<storage, read_write> contact_first_a: array<u32>;
@group(0) @binding(4) var<storage, read_write> contact_first_b: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let value = gid.x;
    if (value >= contact_count[0]) {
        return;
    }
    let key_a = contact_a_body[value];
    if (value == 0u || contact_a_body[value - 1u] != key_a) {
        contact_first_a[key_a] = value + 1u;
    }
    let key_b = contact_keys_b[value];
    if (value == 0u || contact_keys_b[value - 1u] != key_b) {
        contact_first_b[key_b] = value + 1u;
    }
}
