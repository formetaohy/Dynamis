@group(0) @binding(0) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(1) var<storage, read_write> state_scratch: array<BodyState>;
@group(0) @binding(2) var<storage, read> row_moves: array<RowMove>;
@group(0) @binding(3) var<storage, read> fresh_rows: array<BodyState>;
@group(0) @binding(4) var<uniform> params: StepParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.body_move_count) {
        return;
    }
    let entry = row_moves[index];
    if (entry.fresh != NO_SLOT) {
        state_scratch[entry.row] = fresh_rows[entry.fresh];
        return;
    }
    if (entry.source != NO_SLOT) {
        state_scratch[entry.row] = body_states[entry.source];
        return;
    }
    state_scratch[entry.row] = BodyState();
}
