@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> constraints: array<Constraint>;
@group(0) @binding(2) var<storage, read_write> keys_hi: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_lo: array<u32>;
@group(0) @binding(4) var<storage, read_write> values: array<u32>;

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
    keys_hi[index] = constraint.a;
    keys_lo[index] = index;
    values[index] = index;
}
