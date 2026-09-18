@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read> particles: array<SoftParticle>;

fn emit_entry(slot: u32, limit: u32, particle: u32, owner: u32, box: Aabb, level: u32, cells: GridCells, coord: vec3i) {
    if (slot >= limit) {
        counter_add(COUNTER_ENTRY_FAULTS, 1u);
        return;
    }
    let offset = grid_cell_offset(cells, coord);
    var info = particle | (offset << ENTRY_CELL_SHIFT) | (ENTRY_KIND_PARTICLE << ENTRY_KIND_SHIFT) | ENTRY_AWAKE;
    if (offset == 0u) {
        info = info | ENTRY_PRIMARY;
    }
    var entry: GridEntry;
    entry.min = box.min;
    entry.max = box.max;
    entry.group = owner;
    entry.info = info;
    entry_keys[slot] = cell_key(level, coord);
    entry_order[slot] = slot;
    entries[slot] = entry;
}

fn work(index: u32) {
    let particle = particles[index];
    if (particle.owner == NO_BODY) {
        return;
    }
    let box = particle_swept_bounds(particle, params.dt, params.gravity.xyz);
    let base = grid_base_cell();
    let level = grid_entry_level(box, base, ENTRY_REGION_AWAKE);
    if (level > 0u) {
        counter_add(COUNTER_COARSE_ACTIVE, 1u);
    }
    let cells = grid_cells(box, level_cell_size(level, base));
    let count = grid_cell_count(cells);
    let emitted = min(count, MAX_CELLS_PER_COLLIDER);
    if (count > emitted) {
        counter_add(COUNTER_ENTRY_FAULTS, count - emitted);
    }
    let awake_base = counter_load(COUNTER_AWAKE_BASE);
    let limit = arrayLength(&entries);
    for (var ordinal = 0u; ordinal < emitted; ordinal = ordinal + 1u) {
        let slot = awake_base + counter_add(COUNTER_ENTRIES, 1u);
        emit_entry(
            slot,
            limit,
            index,
            particle.owner,
            box,
            level,
            cells,
            grid_cell_at(cells, ordinal),
        );
    }
}
