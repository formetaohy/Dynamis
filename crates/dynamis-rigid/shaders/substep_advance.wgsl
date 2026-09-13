@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> live_bodies: array<u32>;
@group(0) @binding(3) var<storage, read> live_count: array<u32>;

fn extent() -> u32 {
    return min(live_count[0], arrayLength(&live_bodies));
}

fn work(index: u32) {
    let row = live_bodies[index];
    var state = body_states[row];
    state.position = state.position + state.velocity * params.substep_dt;
    let spin = vec4f(state.angular_velocity, 0.0);
    state.orientation = normalize(state.orientation + 0.5 * params.substep_dt * quat_mul(spin, state.orientation));
    body_states[row] = state;
}
