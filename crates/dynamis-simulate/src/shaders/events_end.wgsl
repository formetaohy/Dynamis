@group(0) @binding(0) var<storage, read> prev_contacts: array<Contact>;
@group(0) @binding(8) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(9) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(1) var<storage, read_write> prev_contact_count: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> contacts: array<Contact>;
@group(0) @binding(3) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> events: array<ContactEvent>;
@group(0) @binding(5) var<storage, read_write> event_count: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> spillover: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: SimParams;

fn current_find(key_hi: u32, key_lo: u32) -> bool {
    var lo = 0u;
    var hi = min(atomicLoad(&contact_count[0]), arrayLength(&contacts));
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let a = contacts[mid].a;
        let b = contacts[mid].b;
        if (a < key_hi || (a == key_hi && b < key_lo)) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo < min(atomicLoad(&contact_count[0]), arrayLength(&contacts)) && contacts[lo].a == key_hi && contacts[lo].b == key_lo;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= min(atomicLoad(&prev_contact_count[0]), arrayLength(&prev_contacts))) {
        return;
    }
    let prev = prev_contacts[index];
    if (current_find(prev.a, prev.b) || (prev.events & COLLIDER_EVENT_BEGIN_END) == 0u) {
        return;
    }
    let first_row = prev.a / MAX_COLLIDERS_PER_BODY;
    let second_row = prev.b / MAX_COLLIDERS_PER_BODY;
    let first = body_states[first_row];
    let second = body_states[second_row];
    let held = first.body_id == prev.first_body_id && first.generation == prev.first_generation
        && second.body_id == prev.second_body_id && second.generation == prev.second_generation;
    if (held && !body_is_active(first, body_descs[first_row])
        && !body_is_active(second, body_descs[second_row])) {
        return;
    }
    let slot = atomicAdd(&event_count[0], 1u);
    let segment = arrayLength(&events) / EVENT_SLOTS;
    let base = params.event_slot * segment;
    if (slot < segment) {
        let point = prev.points[0].position;
        events[base + slot] = ContactEvent(EVENT_END, prev.sensor, prev.first_body_id, prev.first_generation, prev.second_body_id, prev.second_generation, point, 0.0, prev.normal, 0.0);
    } else {
        atomicAdd(&spillover[0], 1u);
    }
}
