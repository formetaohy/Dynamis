@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(4) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(5) var<storage, read> constraint_rows: array<ConstraintRows>;
@group(0) @binding(6) var<storage, read> joint_rows: array<u32>;
@group(0) @binding(7) var<storage, read> joint_layers: array<u32>;
@group(0) @binding(8) var<storage, read> joint_components: array<u32>;
@group(0) @binding(9) var<storage, read_write> resolution: array<vec4f>;
@group(0) @binding(10) var<storage, read> joint_batches: array<u32>;

const JOINT_BATCH_WORDS: u32 = 4u;

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
    let batch = wgid.x * JOINT_BATCH_WORDS;
    let lanes = joint_batches[batch + 2u];
    let span = joint_batches[batch + 3u];
    let component = joint_batches[batch] + lid.x / lanes;
    let lane = lid.x % lanes;
    var layer_offset = 0u;
    var layer_count = 0u;
    if (component < joint_batches[batch] + joint_batches[batch + 1u]) {
        layer_offset = joint_components[component * 2u];
        layer_count = joint_components[component * 2u + 1u];
    }
    for (var sweep = 0u; sweep < 2u; sweep = sweep + 1u) {
        for (var step = 0u; step < span; step = step + 1u) {
            if (layer_count > step) {
                let layer = select(layer_offset + step, layer_offset + layer_count - 1u - step, sweep == 1u);
                let first = joint_layers[layer * 2u];
                let count = joint_layers[layer * 2u + 1u];
                for (var index = lane; index < count; index = index + lanes) {
                    project(joint_rows[first + index]);
                }
            }
            if (lanes > 1u) {
                workgroupBarrier();
            }
        }
    }
}
