@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(6) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(7) var<storage, read> body_activity: array<u32>;
@group(0) @binding(8) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> body_descs: array<BodyDescriptor>;

fn work(index: u32) {
    let owner = collider_owners[index];
    if (owner == NO_BODY || body_moves(body_descs[owner])) {
        return;
    }
    let entry = collider_cells(index, ENTRY_REGION_IMMOVABLE);
    let emitted = collider_entry_cost(entry, collider_awake(owner));
    let limit = entry_immovable_base();
    for (var ordinal = 0u; ordinal < emitted; ordinal = ordinal + 1u) {
        let slot = counter_add(COUNTER_IMMOVABLE_ENTRIES, 1u);
        counter_add(COUNTER_IMMOVABLE_EMITTED, 1u);
        emit_collider(
            slot,
            limit,
            index,
            owner,
            false,
            entry,
            grid_cell_at(entry.cells, ordinal),
        );
    }
}
