@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> island_parents: array<u32>;
@group(0) @binding(2) var<storage, read> constraints: array<Constraint>;
@group(0) @binding(3) var<storage, read_write> bucket_hi: array<u32>;
@group(0) @binding(4) var<storage, read_write> bucket_lo: array<u32>;
@group(0) @binding(5) var<storage, read_write> bucket_values: array<u32>;

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
    let root = island_parents[constraint.a];
    bucket_hi[index] = root;
    bucket_lo[index] = index;
    bucket_values[index] = index;
}
