@group(0) @binding(0) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(1) var<storage, read> state_scratch: array<BodyState>;
@group(0) @binding(2) var<storage, read> row_moves: array<RowMove>;
@group(0) @binding(3) var<uniform> params: StepParams;

fn work(index: u32) {
    let row = row_moves[index].row;
    body_states[row] = state_scratch[row];
}
