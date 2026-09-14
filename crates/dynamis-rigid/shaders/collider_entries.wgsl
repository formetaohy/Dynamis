@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(6) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(7) var<storage, read> body_activity: array<u32>;
@group(0) @binding(8) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> body_descs: array<BodyDescriptor>;

fn emit_entry(
    collider: u32,
    owner: u32,
    awake: bool,
    level: u32,
    cells: GridCells,
    coord: vec3i,
) {
    let slot = counter_add(COUNTER_ENTRIES, 1u);
    let offset = grid_cell_offset(cells, coord);
    var info = collider | (offset << ENTRY_CELL_SHIFT) | (ENTRY_KIND_COLLIDER << ENTRY_KIND_SHIFT);
    if (awake) {
        info = info | ENTRY_AWAKE;
    }
    if (body_is_movable(body_descs[owner])) {
        info = info | ENTRY_MOBILE;
    }
    if (offset == 0u) {
        info = info | ENTRY_PRIMARY;
    }
    var entry: GridEntry;
    entry.min = aabbs[collider].min;
    entry.max = aabbs[collider].max;
    entry.group = owner;
    entry.info = info;
    entry_keys[slot] = cell_key(level, coord);
    entry_order[slot] = slot;
    entries[slot] = entry;
}

fn work(index: u32) {
    let owner = collider_owners[index];
    if (owner == NO_BODY) {
        return;
    }
    let aabb = aabbs[index];
    let base = grid_base_cell();
    let level = grid_entry_level(aabb, base);
    let awake = body_activity[owner] != 0u || atomicLoad(&wake_flags[owner]) != 0u;
    if (level > 0u && awake) {
        counter_add(COUNTER_COARSE_ACTIVE, 1u);
    }
    let cells = grid_cells(aabb, level_cell_size(level, base));
    let count = grid_cell_count(cells);
    let emitted = min(count, MAX_CELLS_PER_COLLIDER);
    if (count > emitted) {
        counter_add(COUNTER_ENTRY_FAULTS, count - emitted);
    }
    for (var ordinal = 0u; ordinal < emitted; ordinal = ordinal + 1u) {
        emit_entry(index, owner, awake, level, cells, grid_cell_at(cells, ordinal));
    }
}
