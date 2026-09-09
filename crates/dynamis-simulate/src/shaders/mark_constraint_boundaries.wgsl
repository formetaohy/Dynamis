@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> constraint_keys_a: array<u32>;
@group(0) @binding(2) var<storage, read> constraint_keys_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> constraint_first_a: array<u32>;
@group(0) @binding(4) var<storage, read_write> constraint_first_b: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let value = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (value >= params.constraint_count) {
        return;
    }
    let key_a = constraint_keys_a[value];
    let key_b = constraint_keys_b[value];
    if (key_a == NO_BODY || key_b == NO_BODY) {
        return;
    }
    if (value == 0u || constraint_keys_a[value - 1u] != key_a) {
        constraint_first_a[key_a] = value + 1u;
    }
    if (value == 0u || constraint_keys_b[value - 1u] != key_b) {
        constraint_first_b[key_b] = value + 1u;
    }
}
