@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;

fn work(index: u32) {
    var state = body_states[index];
    state.force = vec3f(0.0);
    state.torque = vec3f(0.0);
    body_states[index] = state;
}
