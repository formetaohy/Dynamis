fn emit_contact_fact(
    kind: u32,
    sensor: u32,
    first_target: u32,
    first_id: u32,
    first_generation: u32,
    second_target: u32,
    second_id: u32,
    second_generation: u32,
    point: vec3f,
    normal: vec3f,
) {
    let slot = atomicAdd(&event_count[0], 1u);
    let segment = arrayLength(&events) / SEGMENT_COUNT;
    if (slot >= segment) {
        atomicAdd(&spillover[0], 1u);
        return;
    }
    events[step_segment() * segment + slot] = ContactEvent(
        point, 0.0, normal, 0.0,
        kind, sensor,
        first_target, first_id, first_generation,
        second_target, second_id, second_generation,
        0u,
    );
}
