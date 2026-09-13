@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read_write> ccd_factor: array<u32>;

fn work(index: u32) {
    ccd_factor[index] = bitcast<u32>(1.0);
    var state = body_states[index];
    if (state.sleeping != 0u) {
        freeze_body(&state);
        body_states[index] = state;
        return;
    }
    state.prev_position = state.position;
    body_states[index] = state;
}
