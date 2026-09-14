@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> block_counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> resolution: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> contributions: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> velocity_deltas: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> position_deltas: array<atomic<u32>>;

fn work(index: u32) {
    block_counts[index] = 0u;
    resolution[index] = vec4f(0.0);
    atomicStore(&contributions[index], 0u);
    for (var word = 0u; word < SOLVER_DELTA_WORDS; word = word + 1u) {
        atomicStore(&velocity_deltas[index * SOLVER_DELTA_WORDS + word], 0u);
        atomicStore(&position_deltas[index * SOLVER_DELTA_WORDS + word], 0u);
    }
}
