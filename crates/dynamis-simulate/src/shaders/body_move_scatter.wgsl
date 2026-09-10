@group(0) @binding(0) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(1) var<storage, read> state_scratch: array<BodyState>;
@group(0) @binding(2) var<storage, read> row_moves: array<RowMove>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.body_move_count) {
        return;
    }
    let row = row_moves[index].row;
    body_states[row] = state_scratch[row];
}
