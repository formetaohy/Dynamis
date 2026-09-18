struct EntryCells {
    level: u32,
    cells: GridCells,
}

fn collider_cells(collider: u32, region: u32) -> EntryCells {
    let base = grid_base_cell();
    let level = grid_entry_level(aabbs[collider], base, region);
    var entry: EntryCells;
    entry.level = level;
    entry.cells = grid_cells(aabbs[collider], level_cell_size(level, base));
    return entry;
}

fn collider_awake(owner: u32) -> bool {
    return body_activity[owner] != 0u || atomicLoad(&wake_flags[owner]) != 0u;
}

fn collider_entry_cost(entry: EntryCells, awake: bool) -> u32 {
    let count = grid_cell_count(entry.cells);
    let emitted = min(count, MAX_CELLS_PER_COLLIDER);
    if (count > emitted) {
        counter_add(COUNTER_ENTRY_FAULTS, count - emitted);
    }
    if (entry.level > 0u && awake) {
        counter_add(COUNTER_COARSE_ACTIVE, 1u);
    }
    return emitted;
}

fn emit_collider(
    slot: u32,
    limit: u32,
    collider: u32,
    owner: u32,
    awake: bool,
    entry: EntryCells,
    coord: vec3i,
) {
    if (slot >= limit) {
        counter_add(COUNTER_ENTRY_FAULTS, 1u);
        return;
    }
    let offset = grid_cell_offset(entry.cells, coord);
    var info = collider | (offset << ENTRY_CELL_SHIFT) | (ENTRY_KIND_COLLIDER << ENTRY_KIND_SHIFT);
    if (awake) {
        info = info | ENTRY_AWAKE;
    }
    if (body_moves(body_descs[owner])) {
        info = info | ENTRY_MOBILE;
    }
    if (offset == 0u) {
        info = info | ENTRY_PRIMARY;
    }
    var record: GridEntry;
    record.min = aabbs[collider].min;
    record.max = aabbs[collider].max;
    record.group = owner;
    record.info = info;
    entry_keys[slot] = cell_key(entry.level, coord);
    entry_order[slot] = slot;
    entries[slot] = record;
}
