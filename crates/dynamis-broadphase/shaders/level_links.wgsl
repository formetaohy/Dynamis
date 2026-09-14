@group(0) @binding(4) var<storage, read_write> pair_major: array<u32>;
@group(0) @binding(5) var<storage, read_write> pair_minor: array<u32>;

fn emit_pair(first: u32, second: u32) {
    let first_info = entries[first].info;
    let second_info = entries[second].info;
    if (entry_kind(first_info) != ENTRY_KIND_COLLIDER || entry_kind(second_info) != ENTRY_KIND_COLLIDER) {
        return;
    }
    if (first == second || entries[first].group == entries[second].group) {
        return;
    }
    if (!entry_mobile(first_info) && !entry_mobile(second_info)) {
        return;
    }
    let slot = counter_add(COUNTER_PAIRS, 1u);
    if (slot < arrayLength(&pair_minor)) {
        let a = entry_index(first_info);
        let b = entry_index(second_info);
        pair_minor[slot] = max(a, b);
        pair_major[slot] = min(a, b);
    } else {
        counter_add(COUNTER_REFUSED_PAIRS, 1u);
    }
}

fn link_level(node: u32, level: u32, awake: bool, live: u32, grid: f32) {
    let box = entry_box(node);
    let cell_size = level_cell_size(level, grid);
    let cells = grid_cells(box, cell_size);
    let count = grid_cell_count(cells);
    for (var ordinal = 0u; ordinal < count; ordinal = ordinal + 1u) {
        let cell = grid_cell_at(cells, ordinal);
        let range = entry_cell_bounds(live, level, cell);
        for (var entry = range.x; entry < range.y; entry = entry + 1u) {
            let other = entry_node(entry);
            if (other == node) {
                continue;
            }
            if (!awake && !entry_awake(entries[other].info)) {
                continue;
            }
            let other_box = entry_box(other);
            if (!aabb_overlaps(box, other_box)) {
                continue;
            }
            if (any(entry_cell(other, cell_size) != cell)) {
                continue;
            }
            if (any(grid_overlap_cell(box, other_box, cell_size) != cell)) {
                continue;
            }
            emit_pair(node, other);
        }
    }
}

fn extent() -> u32 {
    return entry_live();
}

fn work(index: u32) {
    let live = extent();
    if (index >= live) {
        return;
    }
    let node = entry_node(index);
    let info = entries[node].info;
    if (entry_kind(info) != ENTRY_KIND_COLLIDER || !entry_primary(info)) {
        return;
    }
    let awake = entry_awake(info);
    if (!awake && counter_load(COUNTER_COARSE_ACTIVE) == 0u) {
        return;
    }
    let grid = grid_base_cell();
    let level = shape_levels(entry_box(node), grid);
    var coarser = grid_coarser_levels(grid_occupied(), level);
    while (coarser != 0u) {
        let link = 31u - countLeadingZeros(coarser);
        coarser = coarser & ~(1u << link);
        link_level(node, link, awake, live, grid);
    }
}
