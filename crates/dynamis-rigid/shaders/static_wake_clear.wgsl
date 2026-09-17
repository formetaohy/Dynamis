@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;

fn work(index: u32) {
    if (index >= params.dynamic_count && index < params.body_count) {
        if (atomicExchange(&wake_flags[index], 0u) != 0u) {
            counter_add(COUNTER_IMMOVABLE_WOKE, 1u);
        }
    }
}
