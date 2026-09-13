@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> island_parents: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> island_state: array<atomic<u32>>;

fn work(index: u32) {
    atomicStore(&island_parents[index], index);
    atomicStore(&island_state[index], 0u);
}
