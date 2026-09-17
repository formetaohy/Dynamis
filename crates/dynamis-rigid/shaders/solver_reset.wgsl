@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> block_counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> resolution: array<vec4f>;

fn work(index: u32) {
    block_counts[index] = 0u;
    resolution[index] = vec4f(0.0);
}
