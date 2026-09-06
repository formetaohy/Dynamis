@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> constraint_keys_a: array<u32>;
@group(0) @binding(2) var<storage, read> constraint_keys_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> first_a: array<u32>;
@group(0) @binding(4) var<storage, read_write> first_b: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let value = gid.x;
    if (value >= params.constraint_count) {
        return;
    }
    let key_a = constraint_keys_a[value];
    if (value == 0u || constraint_keys_a[value - 1u] != key_a) {
        first_a[key_a] = value + 1u;
    }
    let key_b = constraint_keys_b[value];
    if (value == 0u || constraint_keys_b[value - 1u] != key_b) {
        first_b[key_b] = value + 1u;
    }
}
