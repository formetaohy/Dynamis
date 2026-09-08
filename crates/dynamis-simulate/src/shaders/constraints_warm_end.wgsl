@group(0) @binding(0) var<storage, read_write> constraints: array<Constraint>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= arrayLength(&constraints)) {
        return;
    }
    let constraint = constraints[index];
    if ((constraint.flags & CONSTRAINT_WARM_START) == 0u) {
        constraints[index].accumulated = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
}
