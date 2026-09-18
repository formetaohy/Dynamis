@group(0) @binding(7) var<storage, read_write> wake_flags: array<atomic<u32>>;

fn entry_awake(view: EntryView, position: u32, node: u32) -> bool {
    if (entry_region_of(view, position) == ENTRY_REGION_IMMOVABLE) {
        return atomicLoad(&wake_flags[entry_group(node)]) != 0u;
    }
    return entry_awake_bit(entries[node].info);
}

fn emit_pair(first: u32, second: u32) {
    let first_info = entries[first].info;
    let second_info = entries[second].info;
    if (entry_kind(first_info) != ENTRY_KIND_COLLIDER || entry_kind(second_info) != ENTRY_KIND_COLLIDER) {
        return;
    }
    if (first == second || entries[first].group == entries[second].group) {
        return;
    }
    if (!entry_mobile(first_info) && !entry_mobile(second_info)) {
        return;
    }
    let slot = counter_add(COUNTER_PAIRS, 1u);
    if (slot < arrayLength(&pair_minor)) {
        let a = entry_index(first_info);
        let b = entry_index(second_info);
        pair_minor[slot] = max(a, b);
        pair_major[slot] = min(a, b);
    } else {
        counter_add(COUNTER_REFUSED_PAIRS, 1u);
    }
}
