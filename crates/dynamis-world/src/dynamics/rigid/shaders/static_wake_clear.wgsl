@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> wake_flags: array<atomic<u32>>;

fn work(index: u32) {
    if (index >= params.dynamic_count && index < params.body_count) {
        atomicStore(&wake_flags[index], 0u);
    }
}
