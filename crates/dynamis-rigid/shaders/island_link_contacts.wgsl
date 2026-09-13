@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> island_parents: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read> collider_owners: array<u32>;

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
    return min(atomicLoad(&contact_count[0]), arrayLength(&contacts));
}

fn work(index: u32) {
    let contact = contacts[index];
    if (contact.point_count == 0u || contact.sensor == 1u || !contact_touches(contact, params.slop)) {
        return;
    }
    let first_slot = collider_owners[contact.a];
    let second_slot = collider_owners[contact.b];
    carry_static_wake(first_slot, second_slot);
    carry_static_wake(second_slot, first_slot);
    island_link(first_slot, second_slot);
}
