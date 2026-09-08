@group(0) @binding(0) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(1) var<storage, read_write> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.constraint_count) {
        return;
    }
    if ((constraint_descs[index].flags & CONSTRAINT_WARM_START) != 0u) {
        return;
    }
    var runtime = constraint_runtime[index];
    runtime.accumulated = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    constraint_runtime[index] = runtime;
}
