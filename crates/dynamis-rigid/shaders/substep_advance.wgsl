@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;

fn work(index: u32) {
    var state = body_states[index];
    if (state.sleeping != 0u) {
        return;
    }
    state.position = state.position + state.velocity * params.substep_dt;
    let spin = vec4f(state.angular_velocity, 0.0);
    state.orientation = normalize(state.orientation + 0.5 * params.substep_dt * quat_mul(spin, state.orientation));
    body_states[index] = state;
}
