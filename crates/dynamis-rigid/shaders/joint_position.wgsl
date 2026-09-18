@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(4) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(5) var<storage, read> constraint_rows: array<ConstraintRows>;
@group(0) @binding(6) var<storage, read> joint_rows: array<u32>;
@group(0) @binding(7) var<storage, read> joint_layers: array<u32>;
@group(0) @binding(8) var<storage, read> joint_groups: array<u32>;
@group(0) @binding(9) var<storage, read_write> resolution: array<vec4f>;

fn project(index: u32) {
    let pair = joint_correction(index);
    if (!pair_holds(pair)) {
        return;
    }
    let rows = constraint_rows[index];
    var first = body_states[rows.first_row];
    var second = body_states[rows.second_row];
    first.position = first.position + pair.first.linear;
    first.orientation = normalize(quat_mul(vec4f(pair.first.angular * 0.5, 1.0), first.orientation));
    second.position = second.position + pair.second.linear;
    second.orientation = normalize(quat_mul(vec4f(pair.second.angular * 0.5, 1.0), second.orientation));
    body_states[rows.first_row] = first;
    body_states[rows.second_row] = second;
    resolution[rows.first_row] = vec4f(resolution[rows.first_row].xyz + pair.first.linear, 0.0);
    resolution[rows.second_row] = vec4f(resolution[rows.second_row].xyz + pair.second.linear, 0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let group = wgid.x;
    if (group >= arrayLength(&joint_groups) / 2u) {
        return;
    }
    let layer_offset = joint_groups[group * 2u];
    let layer_count = joint_groups[group * 2u + 1u];
    if (layer_count == 0u) {
        return;
    }
    for (var sweep = 0u; sweep < 2u; sweep = sweep + 1u) {
        for (var step = 0u; step < layer_count; step = step + 1u) {
            let layer = select(layer_offset + step, layer_offset + layer_count - 1u - step, sweep == 1u);
            let first = joint_layers[layer * 2u];
            let count = joint_layers[layer * 2u + 1u];
            for (var index = lid.x; index < count; index = index + WORKGROUP_SIZE) {
                project(joint_rows[first + index]);
            }
            workgroupBarrier();
        }
    }
}
