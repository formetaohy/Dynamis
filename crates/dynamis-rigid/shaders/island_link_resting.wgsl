@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> resting: array<Contact>;
@group(0) @binding(2) var<storage, read> resting_live: array<u32>;
@group(0) @binding(3) var<storage, read_write> resting_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> row_of_body: array<u32>;
@group(0) @binding(5) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(6) var<storage, read_write> island_parents: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> wake_flags: array<atomic<u32>>;

fn resolve_row(body_id: u32, generation: u32) -> u32 {
    if (body_id >= arrayLength(&row_of_body)) {
        return NO_BODY;
    }
    let row = row_of_body[body_id];
    if (row >= arrayLength(&body_states) || !contact_row_matches(body_states[row], body_id, generation)) {
        return NO_BODY;
    }
    return row;
}

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

fn extent() -> u32 {
    return min(atomicLoad(&resting_count[0]), arrayLength(&resting_live));
}

fn work(index: u32) {
    if (resting_live[index] == 0u) {
        return;
    }
    let contact = resting[index];
    let first_row = resolve_row(contact.first_body_id, contact.first_generation);
    let second_row = resolve_row(contact.second_body_id, contact.second_generation);
    if (first_row == NO_BODY || second_row == NO_BODY) {
        return;
    }
    carry_static_wake(first_row, second_row);
    carry_static_wake(second_row, first_row);
    island_link(first_row, second_row);
}
