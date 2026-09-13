@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(2) var<storage, read_write> island_parents: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> constraint_rows: array<ConstraintRows>;

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

fn work(index: u32) {
    if (constraint_runtime[index].broken != 0u) {
        return;
    }
    let rows = constraint_rows[index];
    carry_static_wake(rows.first_row, rows.second_row);
    carry_static_wake(rows.second_row, rows.first_row);
    island_link(rows.first_row, rows.second_row);
}
