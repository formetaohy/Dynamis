@group(0) @binding(0) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(1) var<storage, read_write> constraint_scratch: array<ConstraintRuntime>;
@group(0) @binding(2) var<storage, read> row_moves: array<RowMove>;
@group(0) @binding(3) var<storage, read> fresh_rows: array<ConstraintRuntime>;
@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(6) var<storage, read> row_of_body: array<u32>;
@group(0) @binding(7) var<storage, read> body_states: array<BodyState>;

fn captured_reference(descriptor: ConstraintDescriptor) -> vec4f {
    let first = body_states[row_of_body[descriptor.first_body_id]].orientation;
    let second = body_states[row_of_body[descriptor.second_body_id]].orientation;
    return quat_mul(quat_conjugate(first), second);
}

fn work(index: u32) {
    let entry = row_moves[index];
    if (entry.fresh != NO_SLOT) {
        var runtime = fresh_rows[entry.fresh];
        runtime.reference = captured_reference(constraint_descs[entry.row]);
        constraint_scratch[entry.row] = runtime;
        return;
    }
    if (entry.source != NO_SLOT) {
        constraint_scratch[entry.row] = constraint_runtime[entry.source];
        return;
    }
    constraint_scratch[entry.row] = ConstraintRuntime();
}
