@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_motion: array<u32>;
@group(0) @binding(2) var<storage, read_write> island_parents: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> island_state: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> wake_flags: array<atomic<u32>>;

fn work(index: u32) {
    if (atomicLoad(&wake_flags[index]) == 0u && body_motion[index] == 0u) {
        return;
    }
    let root = atomicLoad(&island_parents[index]);
    atomicMax(&island_state[root], ISLAND_WAKE);
}
