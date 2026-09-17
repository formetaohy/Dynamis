@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> entry_base: array<u32>;

@compute @workgroup_size(1)
fn main() {
    counter_store(COUNTER_IMMOVABLE_ENTRIES, 0u);
    counter_store(COUNTER_IMMOVABLE_LEVELS, 0u);
    counter_store(COUNTER_ENTRY_BASE, entry_base[0]);
}
