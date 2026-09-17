@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read> blocks: array<u32>;
@group(0) @binding(5) var<storage, read_write> velocity_deltas: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> block_counts: array<u32>;
@group(0) @binding(7) var<storage, read_write> block_count: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> target_speeds: array<f32>;
@group(0) @binding(9) var<storage, read> solver_rounds: array<u32>;
@group(0) @binding(10) var<storage, read_write> counters: array<atomic<u32>>;

struct BlockPair {
    first: Body,
    second: Body,
    split_first: Body,
    split_second: Body,
}

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn body_split(body: Body, scale: f32) -> Body {
    var split = body;
    split.desc.inverse_mass = body.desc.inverse_mass * scale;
    var inertia = body.desc.inverse_inertia;
    for (var index = 0u; index < 6u; index = index + 1u) {
        inertia[index] = inertia[index] * scale;
    }
    split.desc.inverse_inertia = inertia;
    return split;
}

fn block_scale(slot: u32) -> f32 {
    return f32(max(block_counts[slot], 1u));
}

fn block_bodies(first_slot: u32, second_slot: u32) -> BlockPair {
    var pair: BlockPair;
    pair.first = load_body(first_slot);
    pair.second = load_body(second_slot);
    pair.split_first = body_split(pair.first, block_scale(first_slot));
    pair.split_second = body_split(pair.second, block_scale(second_slot));
    if (body_is_inert(pair.first)) {
        pair.first = body_frozen(pair.first);
        pair.split_first = body_frozen(pair.split_first);
    }
    if (body_is_inert(pair.second)) {
        pair.second = body_frozen(pair.second);
        pair.split_second = body_frozen(pair.split_second);
    }
    return pair;
}

fn accumulate_velocity(row: u32, linear: vec3f, angular: vec3f) {
    let base = row * SOLVER_DELTA_WORDS;
    atomicAdd(&velocity_deltas[base], solver_word(linear.x, SOLVER_VELOCITY_SCALE));
    atomicAdd(&velocity_deltas[base + 1u], solver_word(linear.y, SOLVER_VELOCITY_SCALE));
    atomicAdd(&velocity_deltas[base + 2u], solver_word(linear.z, SOLVER_VELOCITY_SCALE));
    atomicAdd(&velocity_deltas[base + 3u], solver_word(angular.x, SOLVER_VELOCITY_SCALE));
    atomicAdd(&velocity_deltas[base + 4u], solver_word(angular.y, SOLVER_VELOCITY_SCALE));
    atomicAdd(&velocity_deltas[base + 5u], solver_word(angular.z, SOLVER_VELOCITY_SCALE));
}

fn commit_block(
    _slot: u32,
    first_slot: u32,
    second_slot: u32,
    delta_a: vec3f,
    spin_a: vec3f,
    delta_b: vec3f,
    spin_b: vec3f,
    residual: bool,
) {
    accumulate_velocity(first_slot, delta_a, spin_a);
    accumulate_velocity(second_slot, delta_b, spin_b);
    if (!residual) {
        return;
    }
    counter_max(COUNTER_SOLVE_LINEAR_RESIDUAL, solver_word(max(length(delta_a), length(delta_b)), SOLVER_VELOCITY_SCALE));
    counter_max(COUNTER_SOLVE_ANGULAR_RESIDUAL, solver_word(max(length(spin_a), length(spin_b)), SOLVER_VELOCITY_SCALE));
}

fn residual_round() -> bool {
    return solver_rounds[0] == params.solve_iterations;
}

fn work(index: u32) {
    if (index >= arrayLength(&blocks) / 2u) {
        return;
    }
    solve_contact_block(index, index, residual_round());
}

fn warm_start(index: u32) {
    if (index >= arrayLength(&blocks) / 2u) {
        return;
    }
    warm_contact_block(index, index);
}
