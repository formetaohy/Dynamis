@group(0) @binding(3) var<uniform> params: StepParams;
@group(0) @binding(4) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(5) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(6) var<storage, read> body_activity: array<u32>;

fn emit_entry(collider: u32, level: u32, coord: vec3i) {
    let slot = counter_add(COUNTER_ENTRIES, 1u);
    if (slot < arrayLength(&entry_keys)) {
        entry_keys[slot] = cell_key(level, coord);
        entry_colliders[slot] = collider;
    } else {
        counter_add(COUNTER_SPILLOVER_ENTRIES, 1u);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let collider_index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (collider_index >= params.collider_count) {
        return;
    }
    let owner = collider_owners[collider_index];
    if (owner == NO_BODY) {
        return;
    }
    let aabb = aabbs[collider_index];
    let cell = grid_base_cell();
    let level = shape_levels(aabb, cell);
    counter_or(COUNTER_GRID_LEVELS, 1u << level);
    if (level > 0u && body_activity[owner] != 0u) {
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
                cells = cells + 1u;
                emit_entry(collider_index, level, vec3i(dx, dy, dz));
            }
        }
    }
}
