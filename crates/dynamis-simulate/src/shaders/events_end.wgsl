@group(0) @binding(0) var<storage, read> prev_contacts: array<Contact>;
@group(0) @binding(1) var<storage, read> prev_contact_count: array<u32>;
@group(0) @binding(2) var<storage, read> contacts: array<Contact>;
@group(0) @binding(3) var<storage, read> contact_count: array<u32>;
@group(0) @binding(4) var<storage, read_write> events: array<ContactEvent>;
@group(0) @binding(5) var<storage, read_write> event_count: atomic<u32>;
@group(0) @binding(6) var<storage, read_write> overflow: array<atomic<u32>>;

fn current_find(key_hi: u32, key_lo: u32) -> bool {
    var lo = 0u;
    var hi = contact_count[0];
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
    return lo < contact_count[0] && contacts[lo].a == key_hi && contacts[lo].b == key_lo;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= prev_contact_count[0]) {
        return;
    }
    let prev = prev_contacts[index];
    if (current_find(prev.a, prev.b) || (prev.events & COLLIDER_EVENT_BEGIN_END) == 0u) {
        return;
    }
    let slot = atomicAdd(&event_count, 1u);
    if (slot < arrayLength(&events)) {
        let point = prev.points[0].position;
        events[slot] = ContactEvent(EVENT_END, prev.sensor, prev.first_body_id, prev.first_generation, prev.second_body_id, prev.second_generation, point, 0.0, prev.normal, 0.0);
    } else {
        atomicAdd(&overflow[1u], 1u);
    }
}
