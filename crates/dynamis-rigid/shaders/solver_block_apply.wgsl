@group(0) @binding(0) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(1) var<storage, read_write> velocity_deltas: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> solver_rows: array<u32>;
@group(0) @binding(3) var<storage, read_write> solver_row_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> solver_rounds: array<atomic<u32>>;

fn take(row: u32, word: u32) -> f32 {
    return solver_value(atomicExchange(&velocity_deltas[row * SOLVER_DELTA_WORDS + word], 0u), SOLVER_VELOCITY_SCALE);
}

fn count_round() {
    atomicAdd(&solver_rounds[0], 1u);
}

fn work(index: u32) {
    if (index == 0u) {
        count_round();
    }
    let row = solver_rows[index];
    let linear = vec3f(take(row, 0u), take(row, 1u), take(row, 2u));
    let angular = vec3f(take(row, 3u), take(row, 4u), take(row, 5u));
    if (all(linear == vec3f(0.0)) && all(angular == vec3f(0.0))) {
        return;
    }
    var body = body_states[row];
    body.velocity = body.velocity + linear;
    body.angular_velocity = body.angular_velocity + angular;
    body_states[row] = body;
}
