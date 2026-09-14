@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let slot = lid.x;
    if (slot == COUNTER_ENTRIES ||
        slot == COUNTER_PAIRS ||
        slot == COUNTER_GRID_LEVELS ||
        slot == COUNTER_CONTACTS ||
        slot == COUNTER_JOINTS ||
        slot == COUNTER_EVENTS ||
        slot == COUNTER_REFUSED_PAIRS ||
        slot == COUNTER_REFUSED_CONTACTS ||
        slot == COUNTER_REFUSED_EVENTS ||
        slot == COUNTER_ENTRY_FAULTS ||
        slot == COUNTER_REFUSED_RESTING ||
        slot == COUNTER_ACTIVE ||
        slot == COUNTER_SLEPT ||
        slot == COUNTER_WOKE ||
        slot == COUNTER_WOKE_DEFERRED ||
        slot == COUNTER_RESTING_GATHER ||
        slot == COUNTER_COARSE_ACTIVE ||
        slot == COUNTER_GRID_SCALE ||
        slot == COUNTER_GRID_EXTENT ||
        slot == COUNTER_PARTICLE_REACH ||
        slot == COUNTER_COARSE_NEIGHBOURS ||
        slot == COUNTER_LIVE ||
        slot == COUNTER_LIVE_FAULTS ||
        slot == COUNTER_SOFT_ACTIVE ||
        slot == COUNTER_SOFT_SLEPT ||
        slot == COUNTER_SOFT_WOKE ||
        slot == COUNTER_BREAKS) {
        atomicStore(&counters[slot * COUNTER_STRIDE_WORDS], 0u);
    }
}
