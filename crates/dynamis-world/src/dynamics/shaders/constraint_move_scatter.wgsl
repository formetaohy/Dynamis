@group(0) @binding(0) var<storage, read_write> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(1) var<storage, read> constraint_scratch: array<ConstraintRuntime>;
@group(0) @binding(2) var<storage, read> row_moves: array<RowMove>;
@group(0) @binding(3) var<uniform> params: StepParams;

fn work(index: u32) {
    let row = row_moves[index].row;
    constraint_runtime[row] = constraint_scratch[row];
}
