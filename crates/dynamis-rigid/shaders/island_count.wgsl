@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> island_parents: array<u32>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;

fn work(index: u32) {
    if (island_parents[index] == index) {
        counter_add(COUNTER_ISLANDS, 1u);
    }
}
