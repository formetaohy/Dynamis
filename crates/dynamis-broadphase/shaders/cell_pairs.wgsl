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
    let slot = counter_add(COUNTER_PAIRS, 1u);
    if (slot < arrayLength(&pair_minor)) {
        let a = entry_index(first_info);
        let b = entry_index(second_info);
        pair_minor[slot] = max(a, b);
        pair_major[slot] = min(a, b);
    } else {
        counter_add(COUNTER_SPILLOVER_PAIRS, 1u);
    }
}

fn emit_cell_mate(first: u32, second: u32, cell_size: f32) {
    if (!aabb_overlaps(entry_box(first), entry_box(second))) {
        return;
    }
    let cell = entry_cell(first, cell_size);
    if (any(entry_cell(second, cell_size) != cell)) {
        return;
    }
    let overlap_min = max(entries[first].min, entries[second].min);
    if (any(vec3i(floor(overlap_min / cell_size)) != cell)) {
        return;
    }
    emit_pair(first, second);
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
    if (entry_kind(info) != ENTRY_KIND_COLLIDER || !entry_awake(info)) {
        return;
    }
    let grid = grid_base_cell();
    let cell_size = level_cell_size(shape_levels(entry_box(node), grid), grid);
    let key = entry_keys[index];
    var cursor = index + 1u;
    while (cursor < live && entry_keys[cursor] == key) {
        emit_cell_mate(node, entry_node(cursor), cell_size);
        cursor = cursor + 1u;
    }
    var back = index;
    while (back > 0u && entry_keys[back - 1u] == key) {
        back = back - 1u;
        let other = entry_node(back);
        if (!entry_awake(entries[other].info)) {
            emit_cell_mate(other, node, cell_size);
        }
    }
}
