@group(0) @binding(0) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read> prev_contacts: array<Contact>;
@group(0) @binding(2) var<storage, read_write> prev_contact_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> events: array<ContactEvent>;
@group(0) @binding(5) var<storage, read_write> event_count: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> spillover: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: SimParams;
@group(0) @binding(8) var<storage, read> resting: array<Contact>;
@group(0) @binding(9) var<storage, read> resting_live: array<u32>;
@group(0) @binding(10) var<storage, read> resting_major: array<u32>;
@group(0) @binding(11) var<storage, read> resting_minor: array<u32>;
@group(0) @binding(12) var<storage, read> resting_slots: array<u32>;
@group(0) @binding(13) var<storage, read_write> resting_index: array<atomic<u32>>;

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

fn holds_same_colliders(held: Contact, contact: Contact) -> bool {
    return held.first_body_id == contact.first_body_id
        && held.first_generation == contact.first_generation
        && held.second_body_id == contact.second_body_id
        && held.second_generation == contact.second_generation
        && held.a % MAX_COLLIDERS_PER_BODY == contact.a % MAX_COLLIDERS_PER_BODY
        && held.b % MAX_COLLIDERS_PER_BODY == contact.b % MAX_COLLIDERS_PER_BODY;
}

fn resting_find(contact: Contact) -> u32 {
    let count = min(atomicLoad(&resting_index[0]), arrayLength(&resting_slots));
    var lo = 0u;
    var hi = count;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let major = resting_major[mid];
        if (major < contact.first_body_id || (major == contact.first_body_id && resting_minor[mid] < contact.second_body_id)) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    while (lo < count && resting_major[lo] == contact.first_body_id && resting_minor[lo] == contact.second_body_id) {
        let slot = resting_slots[lo];
        if (slot < arrayLength(&resting_live) && resting_live[slot] == 1u) {
            let held = resting[slot];
            if (holds_same_colliders(held, contact)) {
                return slot;
            }
        }
        lo = lo + 1u;
    }
    return NO_SLOT;
}

fn emit_event(kind: u32, sensor: u32, first_id: u32, first_generation: u32, second_id: u32, second_generation: u32, point: vec3f, normal: vec3f) {
    let slot = atomicAdd(&event_count[0], 1u);
    let segment = arrayLength(&events) / EVENT_SLOTS;
    let base = params.event_slot * segment;
    if (slot < segment) {
        events[base + slot] = ContactEvent(kind, sensor, first_id, first_generation, second_id, second_generation, point, 0.0, normal, 0.0);
    } else {
        atomicAdd(&spillover[0], 1u);
    }
}

fn carries_over(contact: Contact, held: Contact) -> bool {
    return dot(contact.normal, held.normal) >= NORMAL_MATCH;
}

fn relay_impulses(contact: Contact, held: Contact) -> Contact {
    var relayed = contact;
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        relayed.points[point_index].accumulated_normal = held.points[point_index].accumulated_normal;
        relayed.points[point_index].accumulated_tangent_1 = held.points[point_index].accumulated_tangent_1;
        relayed.points[point_index].accumulated_tangent_2 = held.points[point_index].accumulated_tangent_2;
    }
    return relayed;
}

fn announce(flag: u32, kind: u32, contact: Contact) {
    if ((contact.events & flag) == 0u) {
        return;
    }
    emit_event(kind, contact.sensor, contact.first_body_id, contact.first_generation, contact.second_body_id, contact.second_generation, contact.points[0].position, contact.normal);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= min(atomicLoad(&contact_count[0]), arrayLength(&contacts))) {
        return;
    }
    let contact = contacts[index];
    let prev_slot = prev_find(contact.a, contact.b);
    if (prev_slot != NO_BODY) {
        let held = prev_contacts[prev_slot];
        if (!carries_over(contact, held)) {
            return;
        }
        announce(COLLIDER_EVENT_PERSIST, EVENT_PERSIST, contact);
        contacts[index] = relay_impulses(contact, held);
        return;
    }
    let resting_slot = resting_find(contact);
    if (resting_slot == NO_SLOT) {
        announce(COLLIDER_EVENT_BEGIN_END, EVENT_BEGIN, contact);
        return;
    }
    let held = resting[resting_slot];
    if (!carries_over(contact, held)) {
        return;
    }
    announce(COLLIDER_EVENT_PERSIST, EVENT_PERSIST, contact);
    contacts[index] = relay_impulses(contact, held);
}
