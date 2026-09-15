@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> contacts: array<Contact>;
@group(0) @binding(2) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> colliders: array<Collider>;
@group(0) @binding(4) var<storage, read_write> impacts: array<ImpactEvent>;
@group(0) @binding(5) var<storage, read_write> impact_count: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> spillover: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>>;

fn extent() -> u32 {
    return min(atomicLoad(&contact_count[0]), arrayLength(&contacts));
}

fn work(index: u32) {
    let contact = contacts[index];
    let impact = contact_impact(contact);
    let threshold = min(
        colliders[contact.a].impact_force,
        colliders[contact.b].impact_force,
    );
    if (!(impact.x / params.dt > threshold)) {
        return;
    }
    let segment = arrayLength(&impacts) / EVENT_SLOTS;
    let slot = atomicAdd(&impact_count[0], 1u);
    if (slot >= segment) {
        atomicAdd(&spillover[0], 1u);
        return;
    }
    impacts[step_event_slot() * segment + slot] = ImpactEvent(
        contact.first_body_id,
        contact.first_generation,
        contact.second_body_id,
        contact.second_generation,
        contact.points[0].position,
        impact.x,
        contact.normal,
        impact.y,
    );
}
