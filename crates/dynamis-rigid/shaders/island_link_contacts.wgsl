@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> island_parents: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;



fn work(index: u32) {
    let contact = contacts[index];
    if (contact.point_count == 0u || contact.sensor == 1u || !contact_touches(contact, params.slop)) {
        return;
    }
    let first_slot = collider_owners[contact.a];
    let second_slot = collider_owners[contact.b];
    island_couple(first_slot, second_slot);
}
