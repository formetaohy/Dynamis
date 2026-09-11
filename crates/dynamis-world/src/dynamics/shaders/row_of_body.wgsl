@group(0) @binding(0) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(1) var<storage, read> row_moves: array<RowMove>;
@group(0) @binding(2) var<storage, read_write> row_of_body: array<u32>;
@group(0) @binding(3) var<uniform> params: StepParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.body_move_count) {
        return;
    }
    let row = row_moves[index].row;
    let state = body_states[row];
    if (state.generation == 0u || state.body_id >= arrayLength(&row_of_body)) {
        return;
    }
    row_of_body[state.body_id] = row;
}
