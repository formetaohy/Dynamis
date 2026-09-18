@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main() {
    counter_store(
        COUNTER_AWAKE_BASE,
        counter_load(COUNTER_ENTRY_BASE) + counter_load(COUNTER_RESTING_ENTRIES),
    );
    counter_store(COUNTER_RESTING_SORT, counter_load(COUNTER_RESTING_ENTRIES));
}
