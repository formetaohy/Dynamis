fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn solve_contact_correction(contact_index: u32) {
    let contact = contacts[contact_index];
    if (!contact_block_resolves(contact) || contact.relaxation > 0.0) {
        return;
    }
    let first_row = blocks[contact_index * 2u];
    let second_row = blocks[contact_index * 2u + 1u];
    let first_loaded = load_body(first_row);
    let second_loaded = load_body(second_row);
    var first = first_loaded;
    var second = second_loaded;
    if (body_is_inert(first_loaded)) {
        first = body_frozen(first_loaded);
    }
    if (body_is_inert(second_loaded)) {
        second = body_frozen(second_loaded);
    }
    let normal = contact.normal;
    let resolved = dot(normal, resolution[second_row].xyz - resolution[first_row].xyz);
    var total = vec3f(0.0);
    var contributing = 0u;
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        let error = max(contact.points[point_index].depth - params.slop - resolved, 0.0);
        if (error <= 0.0) {
            continue;
        }
        let k = linear_momentum_mass(first, second);
        if (k == 0.0) {
            continue;
        }
        total = total + normal * (params.relaxation * error / k);
        contributing = contributing + 1u;
    }
    if (contributing == 0u) {
        return;
    }
    let correction = total / f32(contributing);
    var pair = pair_zero();
    pair.first.linear = -correction * first.desc.inverse_mass;
    pair.second.linear = correction * second.desc.inverse_mass;
    accumulate_correction(first_row, second_row, pair);
}
