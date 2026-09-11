fn solve_contact_correction(contact_index: u32, slot: u32) {
    let contact = contacts[contact_index];
    if (!contact_block_resolves(contact)) {
        store_block_correction(slot, vec3f(0.0), vec3f(0.0));
        return;
    }
    let first_loaded = load_body(contact.a / MAX_COLLIDERS_PER_BODY);
    let second_loaded = load_body(contact.b / MAX_COLLIDERS_PER_BODY);
    var first = first_loaded;
    var second = second_loaded;
    if (body_is_inert(first_loaded)) {
        first = body_frozen(first_loaded);
    }
    if (body_is_inert(second_loaded)) {
        second = body_frozen(second_loaded);
    }
    let normal = contact.normal;
    var total = vec3f(0.0);
    var contributing = 0u;
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        let error = max(contact.points[point_index].depth - params.slop, 0.0);
        if (error <= 0.0) {
            continue;
        }
        let position = contact.points[point_index].position;
        let k = point_momentum_mass(first, second, position, position, normal);
        if (k == 0.0) {
            continue;
        }
        total = total + normal * (params.relaxation * error / k);
        contributing = contributing + 1u;
    }
    if (contributing == 0u) {
        store_block_correction(slot, vec3f(0.0), vec3f(0.0));
        return;
    }
    let correction = total / f32(contributing);
    store_block_correction(
        slot,
        -correction * first.desc.inverse_mass,
        correction * second.desc.inverse_mass,
    );
}
