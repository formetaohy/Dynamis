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
        slot == COUNTER_SPILLOVER_PAIRS ||
        slot == COUNTER_SPILLOVER_EVENTS ||
        slot == COUNTER_SPILLOVER_ENTRIES ||
        slot == COUNTER_SPILLOVER_RESTING ||
        slot == COUNTER_ACTIVE ||
        slot == COUNTER_SLEPT ||
        slot == COUNTER_WOKE ||
        slot == COUNTER_WOKE_DEFERRED ||
        slot == COUNTER_RESTING_GATHER ||
        slot == COUNTER_COARSE_ACTIVE ||
        slot == COUNTER_GRID_SCALE ||
        slot == COUNTER_GRID_EXTENT ||
        slot == COUNTER_CLASS_CONFLICTS) {
        atomicStore(&counters[slot * COUNTER_STRIDE_WORDS], 0u);
    }
}
