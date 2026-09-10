@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let slot = lid.x;
    if (slot == COUNTER_ENTRIES ||
        slot == COUNTER_PAIRS ||
        slot == COUNTER_LARGE ||
        slot == COUNTER_CONTACTS ||
        slot == COUNTER_JOINTS ||
        slot == COUNTER_EVENTS ||
        slot == COUNTER_SPILLOVER_PAIRS ||
        slot == COUNTER_SPILLOVER_EVENTS ||
        slot == COUNTER_SPILLOVER_ENTRIES) {
        atomicStore(&counters[slot * COUNTER_STRIDE_WORDS], 0u);
    }
}
