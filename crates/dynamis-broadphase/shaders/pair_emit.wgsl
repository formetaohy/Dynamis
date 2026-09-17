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
