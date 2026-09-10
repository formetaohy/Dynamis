@group(0) @binding(0) var<storage, read_write> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(1) var<storage, read> constraint_scratch: array<ConstraintRuntime>;
@group(0) @binding(2) var<storage, read> row_moves: array<RowMove>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.constraint_move_count) {
        return;
    }
    let row = row_moves[index].row;
    constraint_runtime[row] = constraint_scratch[row];
}
