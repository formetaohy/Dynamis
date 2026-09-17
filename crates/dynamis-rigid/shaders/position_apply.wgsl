@group(0) @binding(0) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(1) var<storage, read_write> position_deltas: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> contributions: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> resolution: array<vec4f>;
@group(0) @binding(4) var<storage, read> live_bodies: array<u32>;
@group(0) @binding(5) var<storage, read_write> live_count: array<atomic<u32>>;

fn take(row: u32, word: u32) -> f32 {
    return solver_value(atomicExchange(&position_deltas[row * SOLVER_DELTA_WORDS + word], 0u), SOLVER_POSITION_SCALE);
}

fn work(index: u32) {
    let row = live_bodies[index];
    let contributing = atomicExchange(&contributions[row], 0u);
    let linear = vec3f(take(row, 0u), take(row, 1u), take(row, 2u));
    let angular = vec3f(take(row, 3u), take(row, 4u), take(row, 5u));
    if (contributing == 0u) {
        return;
    }
    let applied_linear = linear / f32(contributing);
    let applied_angular = angular / f32(contributing);
    var body = body_states[row];
    body.position = body.position + applied_linear;
    body.orientation = normalize(quat_mul(vec4f(applied_angular * 0.5, 1.0), body.orientation));
    resolution[row] = vec4f(resolution[row].xyz + applied_linear, 0.0);
    body_states[row] = body;
}
