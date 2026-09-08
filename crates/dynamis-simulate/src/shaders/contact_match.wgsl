@group(0) @binding(0) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read> prev_contacts: array<Contact>;
@group(0) @binding(2) var<storage, read> prev_contact_count: array<u32>;
@group(0) @binding(3) var<storage, read> contact_count: array<u32>;
@group(0) @binding(4) var<storage, read_write> events: array<ContactEvent>;
@group(0) @binding(5) var<storage, read_write> event_count: atomic<u32>;
@group(0) @binding(6) var<storage, read_write> overflow: array<atomic<u32>>;

const NORMAL_MATCH: f32 = 0.7;

fn prev_find(key_hi: u32, key_lo: u32) -> u32 {
    var lo = 0u;
    var hi = prev_contact_count[0];
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let candidate = prev_contacts[mid];
        if (candidate.a < key_hi || (candidate.a == key_hi && candidate.b < key_lo)) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo < prev_contact_count[0]) {
        let candidate = prev_contacts[lo];
        if (candidate.a == key_hi && candidate.b == key_lo) {
            return lo;
        }
    }
    return NO_BODY;
}

fn emit_event(kind: u32, sensor: u32, first_id: u32, first_generation: u32, second_id: u32, second_generation: u32, point: vec3f, normal: vec3f) {
    let slot = atomicAdd(&event_count, 1u);
    if (slot < arrayLength(&events)) {
        events[slot] = ContactEvent(kind, sensor, first_id, first_generation, second_id, second_generation, point, 0.0, normal, 0.0);
    } else {
        atomicAdd(&overflow[1u], 1u);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= contact_count[0]) {
        return;
    }
    let prev_slot = prev_find(contacts[index].a, contacts[index].b);
    var contact = contacts[index];
    if (prev_slot == NO_BODY) {
        let point = contact.points[0].position;
        emit_event(EVENT_BEGIN, contact.sensor, contact.first_body_id, contact.first_generation, contact.second_body_id, contact.second_generation, point, contact.normal);
        return;
    }
    let prev = prev_contacts[prev_slot];
    if (dot(contact.normal, prev.normal) < NORMAL_MATCH) {
        return;
    }
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        contact.points[point_index].accumulated_normal = prev.points[point_index].accumulated_normal;
        contact.points[point_index].accumulated_tangent_1 = prev.points[point_index].accumulated_tangent_1;
        contact.points[point_index].accumulated_tangent_2 = prev.points[point_index].accumulated_tangent_2;
    }
    contacts[index] = contact;
}
