fn contact_row_matches(state: BodyState, body_id: u32, generation: u32) -> bool {
    return state.body_id == body_id && state.generation == generation;
}

fn contact_pair_key(contact: Contact) -> vec2u {
    return vec2u(
        min(contact.first_body_id, contact.second_body_id),
        max(contact.first_body_id, contact.second_body_id),
    );
}

fn contact_touches(contact: Contact, threshold: f32) -> bool {
    for (var index = 0u; index < contact.point_count; index = index + 1u) {
        if (contact.points[index].depth >= -threshold) {
            return true;
        }
    }
    return false;
}

fn contact_same_roles(held: Contact, current: Contact) -> bool {
    return held.first_body_id == current.first_body_id
        && held.first_generation == current.first_generation
        && held.second_body_id == current.second_body_id
        && held.second_generation == current.second_generation
        && held.a == current.a
        && held.b == current.b;
}

fn contact_same_pair(held: Contact, current: Contact) -> bool {
    let flipped = held.first_body_id == current.second_body_id
        && held.first_generation == current.second_generation
        && held.second_body_id == current.first_body_id
        && held.second_generation == current.first_generation
        && held.a == current.b
        && held.b == current.a;
    return contact_same_roles(held, current) || flipped;
}

fn contact_carries_over(held: Contact, current: Contact) -> bool {
    return contact_same_pair(held, current);
}

fn feature_carries_over(held: u32, current: u32) -> bool {
    return held == current || held == feature_mirror(current);
}

fn contact_relay_impulses(current: Contact, held: Contact) -> Contact {
    var relayed = current;
    for (var point_index = 0u; point_index < current.point_count; point_index = point_index + 1u) {
        for (var held_index = 0u; held_index < held.point_count; held_index = held_index + 1u) {
            let held_point = held.points[held_index];
            if (!feature_carries_over(held_point.feature, relayed.points[point_index].feature)) {
                continue;
            }
            relayed.points[point_index].accumulated_normal = held_point.accumulated_normal;
            relayed.points[point_index].accumulated_tangent_1 = held_point.accumulated_tangent_1;
            relayed.points[point_index].accumulated_tangent_2 = held_point.accumulated_tangent_2;
            break;
        }
    }
    return relayed;
}
