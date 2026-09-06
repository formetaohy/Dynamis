@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> contact_first_a: array<u32>;
@group(0) @binding(2) var<storage, read_write> contact_first_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> constraint_first_a: array<u32>;
@group(0) @binding(4) var<storage, read_write> constraint_first_b: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.body_count) {
        return;
    }
    contact_first_a[index] = 0u;
    contact_first_b[index] = 0u;
    constraint_first_a[index] = 0u;
    constraint_first_b[index] = 0u;
}
