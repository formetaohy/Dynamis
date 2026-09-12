@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read_write> reactions: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> wake_flags: array<atomic<u32>>;

const REACTION_SCALE: f32 = 65536.0;
const REACTION_WORDS: u32 = 8u;

fn word_value(word: u32) -> f32 {
    return f32(i32(word)) / REACTION_SCALE;
}

fn work(index: u32) {
    let base = index * REACTION_WORDS;
    let shift_x = atomicExchange(&reactions[base], 0u);
    let shift_y = atomicExchange(&reactions[base + 1u], 0u);
    let shift_z = atomicExchange(&reactions[base + 2u], 0u);
    let spin_x = atomicExchange(&reactions[base + 4u], 0u);
    let spin_y = atomicExchange(&reactions[base + 5u], 0u);
    let spin_z = atomicExchange(&reactions[base + 6u], 0u);
    if (shift_x == 0u && shift_y == 0u && shift_z == 0u && spin_x == 0u && spin_y == 0u && spin_z == 0u) {
        return;
    }
    let shift = vec3f(word_value(shift_x), word_value(shift_y), word_value(shift_z));
    let spin = vec3f(word_value(spin_x), word_value(spin_y), word_value(spin_z));
    var state = body_states[index];
    state.position = state.position + shift;
    state.prev_position = state.prev_position + shift;
    state.velocity = state.velocity + shift / params.dt;
    state.angular_velocity = state.angular_velocity + spin / params.dt;
    state.orientation = normalize(quat_mul(vec4f(spin * 0.5, 1.0), state.orientation));
    state.sleeping = 0u;
    state.sleep_timer = 0.0;
    body_states[index] = state;
    atomicOr(&wake_flags[index], 1u);
}
