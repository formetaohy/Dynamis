@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(2) var<storage, read_write> a_bodies: array<u32>;
@group(0) @binding(3) var<storage, read_write> a_rows: array<u32>;
@group(0) @binding(4) var<storage, read_write> b_bodies: array<u32>;
@group(0) @binding(5) var<storage, read_write> b_rows: array<u32>;
@group(0) @binding(6) var<storage, read> constraint_descs: array<ConstraintDescriptor>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.constraint_count) {
        return;
    }
    if constraint_runtime[index].broken != 0u {
        a_bodies[index] = NO_BODY;
        b_bodies[index] = NO_BODY;
        a_rows[index] = index;
        b_rows[index] = index;
        return;
    }
    let constraint = constraint_descs[index];
    a_bodies[index] = constraint.a;
    a_rows[index] = index;
    b_bodies[index] = constraint.b;
    b_rows[index] = index;
}
