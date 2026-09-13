@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> observed_ids: array<u32>;
@group(0) @binding(2) var<storage, read> row_of_body: array<u32>;
@group(0) @binding(3) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(4) var<storage, read_write> observed_states: array<BodyState>;

fn work(index: u32) {
    observed_states[index] = body_states[row_of_body[observed_ids[index]]];
}
