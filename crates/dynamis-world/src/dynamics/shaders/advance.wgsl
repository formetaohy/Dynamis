@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.dynamic_count) {
        return;
    }
    var state = body_states[index];
    state.position = state.prev_position + state.velocity * params.dt;
    let spin = vec4f(state.angular_velocity, 0.0);
    state.orientation = normalize(state.orientation + 0.5 * params.dt * quat_mul(spin, state.orientation));
    body_states[index] = state;
}
