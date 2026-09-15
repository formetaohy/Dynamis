@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = global_index(gid);
    if (index >= STEP_RESET_COUNT) {
        return;
    }
    atomicStore(&counters[STEP_RESET_SLOTS[index] * COUNTER_STRIDE_WORDS], 0u);
}
