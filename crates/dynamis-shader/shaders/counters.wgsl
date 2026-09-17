fn counter_load(slot: u32) -> u32 {
    return atomicLoad(&counters[slot * COUNTER_STRIDE_WORDS]);
}

fn counter_add(slot: u32, value: u32) -> u32 {
    return atomicAdd(&counters[slot * COUNTER_STRIDE_WORDS], value);
}

fn counter_store(slot: u32, value: u32) {
    atomicStore(&counters[slot * COUNTER_STRIDE_WORDS], value);
}

fn counter_or(slot: u32, value: u32) {
    atomicOr(&counters[slot * COUNTER_STRIDE_WORDS], value);
}

fn counter_max(slot: u32, value: u32) {
    atomicMax(&counters[slot * COUNTER_STRIDE_WORDS], value);
}

fn step_segment() -> u32 {
    return counter_load(COUNTER_STEP) % SEGMENT_COUNT;
}
