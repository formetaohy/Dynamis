@group(0) @binding(4) var<storage, read_write> pair_major: array<u32>;
@group(0) @binding(5) var<storage, read_write> pair_minor: array<u32>;
@group(0) @binding(6) var<storage, read> body_admitted: array<u32>;


fn emit_cell_mate(first: u32, second: u32, cell_size: f32) {
    if (!aabb_overlaps(entry_box(first), entry_box(second))) {
        return;
    }
    let cell = entry_cell(first, cell_size);
    if (any(entry_cell(second, cell_size) != cell)) {
        return;
    }
    if (any(grid_overlap_cell(entry_box(first), entry_box(second), cell_size) != cell)) {
        return;
    }
    emit_pair(first, second);
}

fn work(index: u32) {
    let view = entry_view();
    if (index >= entry_live(view)) {
        return;
    }
    let node = entry_node(view, index);
    let info = entries[node].info;
    if (entry_kind(info) != ENTRY_KIND_COLLIDER || !entry_awake(view, index, node)) {
        return;
    }
    let grid = grid_base_cell();
    let level = shape_levels(entry_box(node), grid);
    let cell_size = level_cell_size(level, grid);
    let ranges = entry_cell_ranges(view, level, entry_cell(node, cell_size));
    for (var region = 0u; region < GRID_REGION_COUNT; region = region + 1u) {
        let range = entry_range(ranges, region);
        for (var entry = range.x; entry < range.y; entry = entry + 1u) {
            if (entry == index) {
                continue;
            }
            let other = entry_node(view, entry);
            if (region == GRID_REGION_RESTING && body_admitted[entry_group(other)] == 0u) {
                continue;
            }
            if (entry_awake(view, entry, other) && entry < index) {
                continue;
            }
            emit_cell_mate(node, other, cell_size);
        }
    }
}
