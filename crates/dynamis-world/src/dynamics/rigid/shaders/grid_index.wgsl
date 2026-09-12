@group(0) @binding(0) var<storage, read_write> entry_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> entry_colliders: array<u32>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;

fn counter_load(slot: u32) -> u32 {
    return atomicLoad(&counters[slot * COUNTER_STRIDE_WORDS]);
}

fn counter_add(slot: u32, value: u32) -> u32 {
    return atomicAdd(&counters[slot * COUNTER_STRIDE_WORDS], value);
}

fn counter_or(slot: u32, value: u32) {
    atomicOr(&counters[slot * COUNTER_STRIDE_WORDS], value);
}

fn entry_live() -> u32 {
    return min(counter_load(COUNTER_ENTRIES), arrayLength(&entry_colliders));
}

fn grid_base_cell() -> f32 {
    return grid_cell_size(counter_load(COUNTER_GRID_SCALE), counter_load(COUNTER_GRID_EXTENT));
}

fn entry_bounds(live: u32, key: u32) -> vec2u {
    var lo = 0u;
    var hi = live;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (entry_keys[mid] < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let first = lo;
    hi = live;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (entry_keys[mid] <= key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return vec2u(first, lo);
}

fn entry_bounds_of(level: u32, coord: vec3i) -> vec2u {
    return entry_bounds(entry_live(), cell_key(level, coord));
}
