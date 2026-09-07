@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> constraints: array<Constraint>;
@group(0) @binding(2) var<storage, read_write> keys_a: array<u32>;
@group(0) @binding(3) var<storage, read_write> values_a: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_b: array<u32>;
@group(0) @binding(5) var<storage, read_write> values_b: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.constraint_count) {
        return;
    }
    let constraint = constraints[index];
    if (constraint.kind == CONSTRAINT_INVALID) {
        return;
    }
    keys_a[index] = constraint.a;
    values_a[index] = index;
    keys_b[index] = constraint.b;
    values_b[index] = index;
}
