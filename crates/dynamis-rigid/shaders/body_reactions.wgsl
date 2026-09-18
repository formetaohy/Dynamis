@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read_write> reactions: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> woke_count: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> deferred_woke_count: array<atomic<u32>>;

fn take(row: u32, word: u32) -> f32 {
    return reaction_value(atomicExchange(&reactions[row * REACTION_WORDS + word], 0u));
}

fn work(index: u32) {
    let shift = vec3f(take(index, 0u), take(index, 1u), take(index, 2u));
    let spin = vec3f(take(index, 3u), take(index, 4u), take(index, 5u));
    if (all(shift == vec3f(0.0)) && all(spin == vec3f(0.0))) {
        return;
    }
    var state = body_states[index];
    if (state.sleeping != 0u) {
        atomicAdd(&woke_count[0], 1u);
        atomicAdd(&deferred_woke_count[0], 1u);
    }
    state.position = state.position + shift;
    state.prev_position = state.prev_position + shift;
    state.velocity = state.velocity + shift / params.dt;
    state.angular_velocity = state.angular_velocity + spin / params.dt;
    state.orientation = normalize(quat_mul(vec4f(spin * 0.5, 1.0), state.orientation));
    state.prev_orientation = normalize(quat_mul(vec4f(spin * 0.5, 1.0), state.prev_orientation));
    state.sleeping = 0u;
    state.sleep_timer = 0.0;
    body_states[index] = state;
    atomicOr(&wake_flags[index], 1u);
}
