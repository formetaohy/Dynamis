fn announce(flag: u32, kind: u32, contact: Contact) {
    if ((contact.events & flag) == 0u) {
        return;
    }
    let slot = atomicAdd(&event_count[0], 1u);
    let segment = arrayLength(&events) / EVENT_SLOTS;
    let base = params.event_slot * segment;
    if (slot < segment) {
        events[base + slot] = ContactEvent(kind, contact.sensor, contact.first_body_id, contact.first_generation, contact.second_body_id, contact.second_generation, contact.points[0].position, 0.0, contact.normal, 0.0);
    } else {
        atomicAdd(&spillover[0], 1u);
    }
}
