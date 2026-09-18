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
    let awake = collider_awake(owner);
    if (body_activity[owner] == 0u && body_admitted[owner] != 0u) {
        return;
    }
    body_admitted[owner] = 0u;
    let entry = collider_cells(index, ENTRY_REGION_AWAKE);
    let emitted = collider_entry_cost(entry, awake);
    let awake_base = counter_load(COUNTER_AWAKE_BASE);
    let limit = arrayLength(&entries);
    for (var ordinal = 0u; ordinal < emitted; ordinal = ordinal + 1u) {
        let slot = awake_base + counter_add(COUNTER_ENTRIES, 1u);
        emit_collider(
            slot,
            limit,
            index,
            owner,
            awake,
            entry,
            grid_cell_at(entry.cells, ordinal),
        );
    }
}
