@group(0) @binding(3) var<uniform> params: StepParams;
@group(0) @binding(4) var<storage, read_write> pair_major: array<u32>;
@group(0) @binding(5) var<storage, read_write> pair_minor: array<u32>;
@group(0) @binding(6) var<storage, read> body_activity: array<u32>;
@group(0) @binding(7) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(8) var<storage, read> aabbs: array<Aabb>;

fn collider_is_awake(collider: u32) -> bool {
    let owner = collider_owners[collider];
    return owner != NO_BODY && body_activity[owner] != 0u;
}

fn emit_pair(first: u32, second: u32) {
    let owner = collider_owners[first];
    if (first == second || owner == collider_owners[second]) {
        return;
    }
    let slot = counter_add(COUNTER_PAIRS, 1u);
    if (slot < arrayLength(&pair_minor)) {
        let a = min(first, second);
        let b = max(first, second);
        pair_minor[slot] = b;
        pair_major[slot] = a;
    } else {
        counter_add(COUNTER_SPILLOVER_PAIRS, 1u);
    }
}

fn emit_cell_mate(first: u32, first_slot: u32, second: u32, second_slot: u32, cell_size: f32) {
    if (!aabb_overlaps(aabbs[first], aabbs[second])) {
        return;
    }
    let cell = entry_cell(first_slot, aabbs[first], cell_size);
    if (any(entry_cell(second_slot, aabbs[second], cell_size) != cell)) {
        return;
    }
    let overlap_min = max(aabbs[first].min, aabbs[second].min);
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
    let collider = entry_collider(index);
    if (!collider_is_awake(collider)) {
        return;
    }
    let aabb = aabbs[collider];
    let grid = grid_base_cell();
    let cell_size = level_cell_size(shape_levels(aabb, grid), grid);
    let key = entry_keys[index];
    var cursor = index + 1u;
    while (cursor < live && entry_keys[cursor] == key) {
        emit_cell_mate(collider, index, entry_collider(cursor), cursor, cell_size);
        cursor = cursor + 1u;
    }
    var back = index;
    while (back > 0u && entry_keys[back - 1u] == key) {
        back = back - 1u;
        let other = entry_collider(back);
        if (!collider_is_awake(other)) {
            emit_cell_mate(other, back, collider, index, cell_size);
        }
    }
}
