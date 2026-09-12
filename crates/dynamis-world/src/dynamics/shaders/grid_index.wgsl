@group(0) @binding(0) var<storage, read_write> entry_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> entry_colliders: array<u32>;
@group(0) @binding(2) var<storage, read_write> entry_count: array<atomic<u32>>;

fn entry_live() -> u32 {
    return min(atomicLoad(&entry_count[0]), arrayLength(&entry_colliders));
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
