@group(0) @binding(0) var<storage, read_write> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(1) var<storage, read> scratch: array<ConstraintRuntime>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.constraint_count) {
        return;
    }
    constraint_runtime[index] = scratch[index];
}
