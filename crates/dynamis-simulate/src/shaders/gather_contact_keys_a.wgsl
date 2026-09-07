@group(0) @binding(0) var<storage, read> contact_count: array<u32>;
@group(0) @binding(1) var<storage, read> contact_keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read> contact_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_hi: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_lo: array<u32>;
@group(0) @binding(5) var<storage, read_write> values: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= contact_count[0]) {
        return;
    }
    keys_hi[index] = contact_keys_hi[index] >> 2u;
    keys_lo[index] = index;
    values[index] = contact_indices[index];
}
