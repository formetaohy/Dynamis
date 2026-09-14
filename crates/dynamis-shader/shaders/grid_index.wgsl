@group(0) @binding(0) var<storage, read_write> entry_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> entry_order: array<u32>;
@group(0) @binding(2) var<storage, read_write> entries: array<GridEntry>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;

fn counter_load(slot: u32) -> u32 {
    return atomicLoad(&counters[slot * COUNTER_STRIDE_WORDS]);
}

fn counter_add(slot: u32, value: u32) -> u32 {
    return atomicAdd(&counters[slot * COUNTER_STRIDE_WORDS], value);
}

fn counter_or(slot: u32, value: u32) {
    atomicOr(&counters[slot * COUNTER_STRIDE_WORDS], value);
}

fn counter_max(slot: u32, value: u32) {
    atomicMax(&counters[slot * COUNTER_STRIDE_WORDS], value);
}

fn entry_live() -> u32 {
    return min(counter_load(COUNTER_ENTRIES), arrayLength(&entries));
}

fn entry_node(sorted: u32) -> u32 {
    return entry_order[sorted];
}

fn entry_info(node: u32) -> u32 {
    return entries[node].info;
}

fn entry_kind(info: u32) -> u32 {
    return (info & ENTRY_KIND_MASK) >> ENTRY_KIND_SHIFT;
}

fn entry_index(info: u32) -> u32 {
    return info & ENTRY_INDEX_MASK;
}

fn entry_awake(info: u32) -> bool {
    return (info & ENTRY_AWAKE) != 0u;
}

fn entry_mobile(info: u32) -> bool {
    return (info & ENTRY_MOBILE) != 0u;
}

fn entry_primary(info: u32) -> bool {
    return (info & ENTRY_PRIMARY) != 0u;
}

fn entry_group(node: u32) -> u32 {
    return entries[node].group;
}

fn entry_box(node: u32) -> Aabb {
    var box: Aabb;
    box.min = entries[node].min;
    box.max = entries[node].max;
    return box;
}

fn entry_cell(node: u32, cell_size: f32) -> vec3i {
    let bits = (entries[node].info & ENTRY_CELL_MASK) >> ENTRY_CELL_SHIFT;
    let offset = vec3i(i32(bits & 1u), i32((bits >> 1u) & 1u), i32((bits >> 2u) & 1u));
    return vec3i(floor(entries[node].min / cell_size)) + offset;
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

fn grid_base_cell() -> f32 {
    return grid_cell_size(counter_load(COUNTER_GRID_SCALE), counter_load(COUNTER_GRID_EXTENT));
}
