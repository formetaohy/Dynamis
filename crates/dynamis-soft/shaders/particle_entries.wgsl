@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read> particles: array<SoftParticle>;

fn emit_entry(particle: u32, owner: u32, box: Aabb, level: u32, coord: vec3i, offset: u32) {
    let slot = counter_add(COUNTER_ENTRIES, 1u);
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
    let cell = grid_base_cell();
    let level = shape_levels(box, cell);
    counter_or(COUNTER_GRID_LEVELS, 1u << level);
    if (level > 0u) {
        counter_add(COUNTER_COARSE_ACTIVE, 1u);
    }
    let cell_size = level_cell_size(level, cell);
    let min_cell = vec3i(floor(box.min / cell_size));
    let max_cell = vec3i(floor(box.max / cell_size));
    var cells = 0u;
    for (var dx = min_cell.x; dx <= max_cell.x; dx = dx + 1) {
        for (var dy = min_cell.y; dy <= max_cell.y; dy = dy + 1) {
            for (var dz = min_cell.z; dz <= max_cell.z; dz = dz + 1) {
                if (cells >= MAX_CELLS_PER_COLLIDER) {
                    counter_add(COUNTER_ENTRY_FAULTS, 1u);
                    continue;
                }
                let offset = u32(dx - min_cell.x)
                    | (u32(dy - min_cell.y) << 1u)
                    | (u32(dz - min_cell.z) << 2u);
                cells = cells + 1u;
                emit_entry(
                    index,
                    particle.owner,
                    box,
                    level,
                    vec3i(dx, dy, dz),
                    offset,
                );
            }
        }
    }
}

