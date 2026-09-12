const NORMAL_MATCH: f32 = 0.7;
const POINT_MATCH_DISTANCE: f32 = 0.05;

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
    return contact_same_roles(held, current) && dot(current.normal, held.normal) >= NORMAL_MATCH;
}

fn nearest_held_point(point: ManifoldPoint, held: Contact, taken: ptr<function, array<bool, CONTACT_MAX_POINTS>>) -> u32 {
    var nearest = NO_SLOT;
    var nearest_distance = POINT_MATCH_DISTANCE;
    for (var index = 0u; index < held.point_count; index = index + 1u) {
        if ((*taken)[index]) {
            continue;
        }
        let span = distance(point.position, held.points[index].position);
        if (span <= nearest_distance) {
            nearest = index;
            nearest_distance = span;
        }
    }
    return nearest;
}

fn contact_relay_impulses(current: Contact, held: Contact) -> Contact {
    var relayed = current;
    var taken: array<bool, CONTACT_MAX_POINTS>;
    for (var index = 0u; index < CONTACT_MAX_POINTS; index = index + 1u) {
        taken[index] = false;
    }
    for (var point_index = 0u; point_index < current.point_count; point_index = point_index + 1u) {
        let nearest = nearest_held_point(relayed.points[point_index], held, &taken);
        if (nearest == NO_SLOT) {
            continue;
        }
        taken[nearest] = true;
        relayed.points[point_index].accumulated_normal = held.points[nearest].accumulated_normal;
        relayed.points[point_index].accumulated_tangent_1 = held.points[nearest].accumulated_tangent_1;
        relayed.points[point_index].accumulated_tangent_2 = held.points[nearest].accumulated_tangent_2;
    }
    return relayed;
}

