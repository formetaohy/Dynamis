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

fn contact_row_key(contact: Contact, first_row: u32, second_row: u32) -> vec2u {
    let first = first_row * MAX_COLLIDERS_PER_BODY + contact.a % MAX_COLLIDERS_PER_BODY;
    let second = second_row * MAX_COLLIDERS_PER_BODY + contact.b % MAX_COLLIDERS_PER_BODY;
    return vec2u(min(first, second), max(first, second));
}

fn contact_same_roles(held: Contact, current: Contact) -> bool {
    return held.first_body_id == current.first_body_id
        && held.first_generation == current.first_generation
        && held.second_body_id == current.second_body_id
        && held.second_generation == current.second_generation
        && held.a % MAX_COLLIDERS_PER_BODY == current.a % MAX_COLLIDERS_PER_BODY
        && held.b % MAX_COLLIDERS_PER_BODY == current.b % MAX_COLLIDERS_PER_BODY;
}

fn contact_same_pair(held: Contact, current: Contact) -> bool {
    let flipped = held.first_body_id == current.second_body_id
        && held.first_generation == current.second_generation
        && held.second_body_id == current.first_body_id
        && held.second_generation == current.first_generation
        && held.a % MAX_COLLIDERS_PER_BODY == current.b % MAX_COLLIDERS_PER_BODY
        && held.b % MAX_COLLIDERS_PER_BODY == current.a % MAX_COLLIDERS_PER_BODY;
    return contact_same_roles(held, current) || flipped;
}

fn contact_carries_over(held: Contact, current: Contact) -> bool {
    return contact_same_roles(held, current) && dot(current.normal, held.normal) >= NORMAL_MATCH;
}

fn contact_point_matches(current: ManifoldPoint, held: ManifoldPoint) -> bool {
    return distance(current.position, held.position) <= POINT_MATCH_DISTANCE;
}

fn contact_relay_impulses(current: Contact, held: Contact) -> Contact {
    var relayed = current;
    let matched_points = min(current.point_count, held.point_count);
    for (var point_index = 0u; point_index < matched_points; point_index = point_index + 1u) {
        if (!contact_point_matches(relayed.points[point_index], held.points[point_index])) {
            continue;
        }
        relayed.points[point_index].accumulated_normal = held.points[point_index].accumulated_normal;
        relayed.points[point_index].accumulated_tangent_1 = held.points[point_index].accumulated_tangent_1;
        relayed.points[point_index].accumulated_tangent_2 = held.points[point_index].accumulated_tangent_2;
    }
    return relayed;
}

