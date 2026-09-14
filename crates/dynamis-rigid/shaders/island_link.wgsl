fn island_link(first: u32, second: u32) {
    if (first >= params.dynamic_count || second >= params.dynamic_count || first == second) {
        return;
    }
    atomicMin(&island_parents[first], second);
    atomicMin(&island_parents[second], first);
}

fn carry_static_wake(movable: u32, partner: u32) {
    if (partner < params.dynamic_count) {
        return;
    }
    if (atomicLoad(&wake_flags[partner]) != 0u) {
        atomicOr(&wake_flags[movable], 1u);
    }
}
