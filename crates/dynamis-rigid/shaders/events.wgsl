fn announce(flag: u32, kind: u32, contact: Contact) {
    if ((contact.events & flag) == 0u) {
        return;
    }
    emit_contact_fact(
        kind,
        contact.sensor,
        scene_target_of(ENTRY_KIND_COLLIDER, contact.a),
        contact.first_body_id,
        contact.first_generation,
        scene_target_of(ENTRY_KIND_COLLIDER, contact.b),
        contact.second_body_id,
        contact.second_generation,
        contact.points[0].position,
        contact.normal,
    );
}
