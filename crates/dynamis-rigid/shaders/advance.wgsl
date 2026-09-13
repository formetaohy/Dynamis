@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read_write> ccd_factor: array<u32>;

fn work(index: u32) {
    ccd_factor[index] = bitcast<u32>(1.0);
    var state = body_states[index];
    state.position = state.prev_position + state.velocity * params.dt;
    let spin = vec4f(state.angular_velocity, 0.0);
    state.orientation = normalize(state.orientation + 0.5 * params.dt * quat_mul(spin, state.orientation));
    body_states[index] = state;
}
