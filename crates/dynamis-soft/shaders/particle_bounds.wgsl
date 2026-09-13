@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> params: StepParams;
@group(0) @binding(2) var<storage, read> particles: array<SoftParticle>;

fn counter_max(slot: u32, value: u32) {
    atomicMax(&counters[slot * COUNTER_STRIDE_WORDS], value);
}

fn work(index: u32) {
    let particle = particles[index];
    if (particle.owner == NO_BODY) {
        return;
    }
    let tight = particle_tight_bounds(particle);
    let swept = particle_swept_bounds(particle, params.dt, params.gravity.xyz);
    counter_max(COUNTER_GRID_SCALE, bitcast<u32>(grid_scale_of(tight)));
    counter_max(COUNTER_GRID_EXTENT, bitcast<u32>(max(grid_extent_of(tight), 0.0)));
    counter_max(
        COUNTER_PARTICLE_REACH,
        bitcast<u32>(max(0.5 * grid_extent_of(swept), 0.0)),
    );
}

