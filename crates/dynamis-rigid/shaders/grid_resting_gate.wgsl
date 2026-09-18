@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main() {
    counter_store(COUNTER_RESTING_ENTRIES, 0u);
    counter_store(COUNTER_RESTING_LEVELS, 0u);
    counter_store(COUNTER_RESTING_REBUILD, 1u);
}
