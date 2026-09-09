@group(0) @binding(0) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read> prev_contacts: array<Contact>;
@group(0) @binding(2) var<storage, read_write> prev_contact_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> events: array<ContactEvent>;
@group(0) @binding(5) var<storage, read_write> event_count: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> spillover: array<atomic<u32>>;

const NORMAL_MATCH: f32 = 0.7;

fn prev_find(key_hi: u32, key_lo: u32) -> u32 {
    var lo = 0u;
    var hi = min(atomicLoad(&prev_contact_count[0]), arrayLength(&prev_contacts));
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let candidate = prev_contacts[mid];
        if (candidate.a < key_hi || (candidate.a == key_hi && candidate.b < key_lo)) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo < min(atomicLoad(&prev_contact_count[0]), arrayLength(&prev_contacts))) {
        let candidate = prev_contacts[lo];
        if (candidate.a == key_hi && candidate.b == key_lo) {
            return lo;
        }
    }
    return NO_BODY;
}

fn emit_event(kind: u32, sensor: u32, first_id: u32, first_generation: u32, second_id: u32, second_generation: u32, point: vec3f, normal: vec3f) {
    let slot = atomicAdd(&event_count[0], 1u);
    if (slot < arrayLength(&events)) {
        events[slot] = ContactEvent(kind, sensor, first_id, first_generation, second_id, second_generation, point, 0.0, normal, 0.0);
    } else {
        atomicAdd(&spillover[0], 1u);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= min(atomicLoad(&contact_count[0]), arrayLength(&contacts))) {
        return;
    }
    let prev_slot = prev_find(contacts[index].a, contacts[index].b);
    var contact = contacts[index];
    if (prev_slot == NO_BODY) {
        if ((contact.events & COLLIDER_EVENT_BEGIN_END) != 0u) {
            let point = contact.points[0].position;
            emit_event(EVENT_BEGIN, contact.sensor, contact.first_body_id, contact.first_generation, contact.second_body_id, contact.second_generation, point, contact.normal);
        }
        return;
    }
    let prev = prev_contacts[prev_slot];
    if (dot(contact.normal, prev.normal) < NORMAL_MATCH) {
        return;
    }
    if ((contact.events & COLLIDER_EVENT_PERSIST) != 0u) {
        let point = contact.points[0].position;
        emit_event(EVENT_PERSIST, contact.sensor, contact.first_body_id, contact.first_generation, contact.second_body_id, contact.second_generation, point, contact.normal);
    }
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        contact.points[point_index].accumulated_normal = prev.points[point_index].accumulated_normal;
        contact.points[point_index].accumulated_tangent_1 = prev.points[point_index].accumulated_tangent_1;
        contact.points[point_index].accumulated_tangent_2 = prev.points[point_index].accumulated_tangent_2;
    }
    contacts[index] = contact;
}
