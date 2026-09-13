@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(6) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(7) var<storage, read> body_activity: array<u32>;

fn emit_entry(collider: u32, owner: u32, awake: bool, level: u32, coord: vec3i, offset: u32) {
    let slot = counter_add(COUNTER_ENTRIES, 1u);
    if (slot < arrayLength(&entries)) {
        var info = collider | (offset << ENTRY_CELL_SHIFT) | (ENTRY_KIND_COLLIDER << ENTRY_KIND_SHIFT);
        if (awake) {
            info = info | ENTRY_AWAKE;
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
    } else {
        counter_add(COUNTER_SPILLOVER_ENTRIES, 1u);
    }
}

fn work(index: u32) {
    let owner = collider_owners[index];
    if (owner == NO_BODY) {
        return;
    }
    let aabb = aabbs[index];
    let cell = grid_base_cell();
    let level = shape_levels(aabb, cell);
    let awake = body_activity[owner] != 0u;
    counter_or(COUNTER_GRID_LEVELS, 1u << level);
    if (level > 0u && awake) {
        counter_add(COUNTER_COARSE_ACTIVE, 1u);
    }
    let cell_size = level_cell_size(level, cell);
    let min_cell = vec3i(floor(aabb.min / cell_size));
    let max_cell = vec3i(floor(aabb.max / cell_size));
    var cells = 0u;
    for (var dx = min_cell.x; dx <= max_cell.x; dx = dx + 1) {
        for (var dy = min_cell.y; dy <= max_cell.y; dy = dy + 1) {
            for (var dz = min_cell.z; dz <= max_cell.z; dz = dz + 1) {
                if (cells >= MAX_CELLS_PER_COLLIDER) {
                    counter_add(COUNTER_SPILLOVER_ENTRIES, 1u);
                    continue;
                }
                let offset = u32(dx - min_cell.x)
                    | (u32(dy - min_cell.y) << 1u)
                    | (u32(dz - min_cell.z) << 2u);
                cells = cells + 1u;
                emit_entry(index, owner, awake, level, vec3i(dx, dy, dz), offset);
            }
        }
    }
}
