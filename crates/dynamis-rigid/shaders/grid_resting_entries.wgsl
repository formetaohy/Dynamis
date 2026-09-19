@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(6) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(7) var<storage, read> body_activity: array<u32>;
@group(0) @binding(8) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(10) var<storage, read_write> body_admitted: array<u32>;

fn work(index: u32) {
    let owner = collider_owners[index];
    if (owner == NO_BODY || !body_moves(body_descs[owner])) {
        return;
    }
    if (body_activity[owner] != 0u) {
        return;
    }
    body_admitted[owner] = 1u;
    let entry = collider_cells(index, GRID_REGION_RESTING);
    let emitted = collider_entry_cost(entry, collider_awake(owner));
    let resting_base = entry_immovable_base();
    let limit = arrayLength(&entries);
    for (var ordinal = 0u; ordinal < emitted; ordinal = ordinal + 1u) {
        let slot = resting_base + counter_add(COUNTER_RESTING_ENTRIES, 1u);
        emit_collider(
            slot,
            limit,
            index,
            owner,
            collider_awake(owner),
            entry,
            grid_cell_at(entry.cells, ordinal),
        );
    }
}
