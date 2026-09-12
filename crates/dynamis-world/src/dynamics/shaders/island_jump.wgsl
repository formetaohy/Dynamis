@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> island_parents: array<atomic<u32>>;

fn work(index: u32) {
    let parent = atomicLoad(&island_parents[index]);
    let grandparent = atomicLoad(&island_parents[parent]);
    island_parents[index] = min(parent, grandparent);
}
