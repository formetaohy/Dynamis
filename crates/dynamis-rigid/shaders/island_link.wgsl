const ISLAND_HOOK_ATTEMPTS: u32 = 256u;

fn island_find(node: u32) -> u32 {
    var current = node;
    var parent = atomicLoad(&island_parents[current]);
    while (parent != current) {
        current = parent;
        parent = atomicLoad(&island_parents[current]);
    }
    return current;
}

fn island_union(first: u32, second: u32) {
    if (first >= params.dynamic_count || second >= params.dynamic_count || first == second) {
        return;
    }
    for (var attempt = 0u; attempt < ISLAND_HOOK_ATTEMPTS; attempt = attempt + 1u) {
        let first_root = island_find(first);
        let second_root = island_find(second);
        if (first_root == second_root) {
            return;
        }
        let lower = min(first_root, second_root);
        let higher = max(first_root, second_root);
        if (atomicCompareExchangeWeak(&island_parents[higher], higher, lower).exchanged) {
            return;
        }
    }
    counter_add(COUNTER_ISLAND_FAULTS, 1u);
}

fn carry_static_wake(movable: u32, partner: u32) {
    if (partner < params.dynamic_count) {
        return;
    }
    if (atomicLoad(&wake_flags[partner]) != 0u) {
        atomicOr(&wake_flags[movable], 1u);
    }
}

fn island_couple(first: u32, second: u32) {
    carry_static_wake(first, second);
    carry_static_wake(second, first);
    island_union(first, second);
}
