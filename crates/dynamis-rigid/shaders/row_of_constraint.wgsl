@group(0) @binding(0) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(1) var<storage, read> row_moves: array<RowMove>;
@group(0) @binding(2) var<storage, read_write> row_of_constraint: array<u32>;
@group(0) @binding(3) var<storage, read> row_streams: RowStreams;

fn work(index: u32) {
    let row = row_moves[index].row;
    let runtime = constraint_runtime[row];
    if (runtime.generation == 0u || runtime.constraint_id >= arrayLength(&row_of_constraint)) {
        return;
    }
    row_of_constraint[runtime.constraint_id] = row;
}
