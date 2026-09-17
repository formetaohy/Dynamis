@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(5) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(6) var<storage, read> segments: array<u32>;
@group(0) @binding(7) var<storage, read_write> position_deltas: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> resolution: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> contributions: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> block_count: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read> constraint_rows: array<ConstraintRows>;
@group(0) @binding(12) var<storage, read> blocks: array<u32>;

fn accumulate_position(row: u32, linear: vec3f, angular: vec3f) {
    let base = row * SOLVER_DELTA_WORDS;
    atomicAdd(&position_deltas[base], solver_word(linear.x, SOLVER_POSITION_SCALE));
    atomicAdd(&position_deltas[base + 1u], solver_word(linear.y, SOLVER_POSITION_SCALE));
    atomicAdd(&position_deltas[base + 2u], solver_word(linear.z, SOLVER_POSITION_SCALE));
    atomicAdd(&position_deltas[base + 3u], solver_word(angular.x, SOLVER_POSITION_SCALE));
    atomicAdd(&position_deltas[base + 4u], solver_word(angular.y, SOLVER_POSITION_SCALE));
    atomicAdd(&position_deltas[base + 5u], solver_word(angular.z, SOLVER_POSITION_SCALE));
}

fn accumulate_correction(first_row: u32, second_row: u32, pair: CorrectionPair) {
    if (!pair_holds(pair)) {
        return;
    }
    atomicAdd(&contributions[first_row], 1u);
    atomicAdd(&contributions[second_row], 1u);
    accumulate_position(first_row, pair.first.linear, pair.first.angular);
    accumulate_position(second_row, pair.second.linear, pair.second.angular);
}

fn work(index: u32) {
    let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
    if (index < contact_blocks) {
        solve_contact_correction(index);
    } else {
        solve_constraint_correction(index - contact_blocks);
    }
}
