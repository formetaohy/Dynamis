@group(0) @binding(0) var<storage, read_write> entry_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> entry_order: array<u32>;
@group(0) @binding(2) var<storage, read_write> entries: array<GridEntry>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;

const GRID_WALK_CELLS: u32 = 4096u;
const GRID_WALK_ENTRIES: u32 = 4096u;

struct GridCells {
    min: vec3i,
    max: vec3i,
}

struct GridSlice {
    level: u32,
    cell_size: f32,
    first: u32,
    end: u32,
    whole: bool,
}

fn cell_hash(coord: vec3i) -> u32 {
    let x = u32(coord.x) * 0x9E3779B9u;
    let y = u32(coord.y) * 0x85EBCA77u;
    let z = u32(coord.z) * 0xC2B2AE3Du;
    return (x ^ y ^ z ^ (x << 7u) ^ (y >> 3u) ^ (z << 11u)) & CELL_HASH_MASK;
}

fn cell_key(level: u32, coord: vec3i) -> u32 {
    return (level << LEVEL_KEY_SHIFT) | cell_hash(coord);
}

fn level_cell_size(level: u32, cell_size: f32) -> f32 {
    return cell_size * f32(1u << level);
}

fn grid_base_cell() -> f32 {
    return grid_cell_size(counter_load(COUNTER_GRID_SCALE), counter_load(COUNTER_GRID_EXTENT));
}

fn grid_cells(box: Aabb, cell_size: f32) -> GridCells {
    var cells: GridCells;
    cells.min = vec3i(floor(box.min / cell_size));
    cells.max = vec3i(floor(box.max / cell_size));
    return cells;
}

fn grid_cell_span(cells: GridCells) -> vec3u {
    return vec3u(cells.max - cells.min + vec3i(1));
}

fn grid_cell_count(cells: GridCells) -> u32 {
    let span = grid_cell_span(cells);
    return span.x * span.y * span.z;
}

fn grid_cell_at(cells: GridCells, ordinal: u32) -> vec3i {
    let span = grid_cell_span(cells);
    let rows = span.y * span.z;
    return cells.min
        + vec3i(
            i32(ordinal / rows),
            i32((ordinal / span.z) % span.y),
            i32(ordinal % span.z),
        );
}

fn grid_cell_offset(cells: GridCells, coord: vec3i) -> u32 {
    let offset = coord - cells.min;
    return u32(offset.x) | (u32(offset.y) << 1u) | (u32(offset.z) << 2u);
}

fn shape_levels(box: Aabb, base: f32) -> u32 {
    var level = 0u;
    loop {
        let span = grid_cell_span(grid_cells(box, level_cell_size(level, base)));
        if (max(max(span.x, span.y), span.z) <= MAX_CELLS_PER_AXIS || level >= GRID_LEVEL_LIMIT) {
            break;
        }
        level = level + 1u;
    }
    return level;
}

fn grid_entry_level(box: Aabb, base: f32) -> u32 {
    let level = shape_levels(box, base);
    counter_or(COUNTER_GRID_LEVELS, 1u << level);
    return level;
}

fn grid_occupied() -> u32 {
    return counter_load(COUNTER_GRID_LEVELS);
}

fn grid_coarser_levels(occupied: u32, level: u32) -> u32 {
    return occupied & ~((1u << min(level + 1u, 31u)) - 1u);
}

fn grid_overlap_cell(box: Aabb, other: Aabb, cell_size: f32) -> vec3i {
    return vec3i(floor(max(box.min, other.min) / cell_size));
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

fn entry_cell_bounds(live: u32, level: u32, coord: vec3i) -> vec2u {
    return entry_bounds(live, cell_key(level, coord));
}

fn entry_level_bounds(live: u32, level: u32) -> vec2u {
    let first = entry_bounds(live, level << LEVEL_KEY_SHIFT).x;
    let end = entry_bounds(live, (level + 1u) << LEVEL_KEY_SHIFT).x;
    return vec2u(first, end);
}

fn grid_scan_end(slice: GridSlice) -> u32 {
    if (!slice.whole) {
        return slice.end;
    }
    return min(slice.end, slice.first + GRID_WALK_ENTRIES);
}

fn grid_whole_slices(box: Aabb) -> u32 {
    let base = grid_base_cell();
    var whole = 0u;
    var occupied = grid_occupied();
    while (occupied != 0u) {
        let cell_size = level_cell_size(countTrailingZeros(occupied), base);
        if (grid_cell_count(grid_cells(box, cell_size)) > GRID_WALK_CELLS) {
            whole = whole + 1u;
        }
        occupied = occupied & (occupied - 1u);
    }
    return whole;
}

fn grid_cell_slices(box: Aabb) -> u32 {
    let base = grid_base_cell();
    var cells = 0u;
    var occupied = grid_occupied();
    while (occupied != 0u) {
        let cell_size = level_cell_size(countTrailingZeros(occupied), base);
        let count = grid_cell_count(grid_cells(box, cell_size));
        if (count <= GRID_WALK_CELLS) {
            cells = cells + count;
        }
        occupied = occupied & (occupied - 1u);
    }
    return cells;
}

fn grid_whole_slice(box: Aabb, index: u32) -> GridSlice {
    let base = grid_base_cell();
    let live = entry_live();
    var remaining = index;
    var occupied = grid_occupied();
    while (occupied != 0u) {
        let level = countTrailingZeros(occupied);
        occupied = occupied & (occupied - 1u);
        let cell_size = level_cell_size(level, base);
        if (grid_cell_count(grid_cells(box, cell_size)) <= GRID_WALK_CELLS) {
            continue;
        }
        if (remaining == 0u) {
            let range = entry_level_bounds(live, level);
            return GridSlice(level, cell_size, range.x, range.y, true);
        }
        remaining = remaining - 1u;
    }
    return GridSlice(0u, base, 0u, 0u, true);
}

fn grid_cell_slice(box: Aabb, index: u32) -> GridSlice {
    let base = grid_base_cell();
    let live = entry_live();
    var remaining = index;
    var occupied = grid_occupied();
    while (occupied != 0u) {
        let level = countTrailingZeros(occupied);
        occupied = occupied & (occupied - 1u);
        let cell_size = level_cell_size(level, base);
        let cells = grid_cells(box, cell_size);
        let count = grid_cell_count(cells);
        if (count > GRID_WALK_CELLS) {
            continue;
        }
        if (remaining < count) {
            let range = entry_cell_bounds(live, level, grid_cell_at(cells, remaining));
            return GridSlice(level, cell_size, range.x, range.y, false);
        }
        remaining = remaining - count;
    }
    return GridSlice(0u, base, 0u, 0u, false);
}

fn grid_slices(box: Aabb) -> u32 {
    return grid_whole_slices(box) + grid_cell_slices(box);
}

fn grid_slice(box: Aabb, index: u32) -> GridSlice {
    let whole = grid_whole_slices(box);
    if (index < whole) {
        return grid_whole_slice(box, index);
    }
    return grid_cell_slice(box, index - whole);
}
