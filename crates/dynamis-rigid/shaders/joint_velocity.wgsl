@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(4) var<storage, read_write> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(5) var<storage, read> constraint_rows: array<ConstraintRows>;
@group(0) @binding(6) var<storage, read> joint_rows: array<u32>;
@group(0) @binding(7) var<storage, read> joint_layers: array<u32>;
@group(0) @binding(8) var<storage, read> joint_components: array<u32>;
@group(0) @binding(9) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read> solver_rounds: array<u32>;
@group(0) @binding(11) var<storage, read> joint_batches: array<u32>;

const JOINT_BATCH_WORDS: u32 = 4u;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn residual_round() -> bool {
    return solver_rounds[0] == params.solve_iterations;
}

fn walk(batch_index: u32, lid: u32, warming: bool) {
    let batch = batch_index * JOINT_BATCH_WORDS;
    let lanes = joint_batches[batch + 2u];
    let span = joint_batches[batch + 3u];
    let component = joint_batches[batch] + lid / lanes;
    let lane = lid % lanes;
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
                    let row = joint_rows[first + index];
                    if (warming) {
                        warm_joint(row);
                    } else {
                        solve_joint(row, residual_round());
                    }
                }
            }
            if (lanes > 1u) {
                workgroupBarrier();
            }
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    walk(wgid.x, lid.x, false);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn warm(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    walk(wgid.x, lid.x, true);
}
