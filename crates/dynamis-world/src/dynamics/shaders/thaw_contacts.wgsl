@group(0) @binding(0) var<storage, read> resting: array<Contact>;
@group(0) @binding(1) var<storage, read_write> resting_live: array<u32>;
@group(0) @binding(2) var<storage, read_write> resting_next: array<u32>;
@group(0) @binding(3) var<storage, read_write> resting_free: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> resting_count: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(6) var<storage, read> row_of_body: array<u32>;
@group(0) @binding(7) var<storage, read> body_activity: array<u32>;
@group(0) @binding(8) var<storage, read> contacts: array<Contact>;
@group(0) @binding(9) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> events: array<ContactEvent>;
@group(0) @binding(11) var<storage, read_write> event_count: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> spillover: array<atomic<u32>>;
@group(0) @binding(13) var<uniform> params: StepParams;

fn resolve_row(body_id: u32, generation: u32) -> u32 {
    if (body_id >= arrayLength(&row_of_body)) {
        return NO_BODY;
    }
    let row = row_of_body[body_id];
    if (row >= arrayLength(&body_states) || !contact_row_matches(body_states[row], body_id, generation)) {
        return NO_BODY;
    }
    return row;
}

fn current_holds(key_hi: u32, key_lo: u32) -> bool {
    let count = min(atomicLoad(&contact_count[0]), arrayLength(&contacts));
    var lo = 0u;
    var hi = count;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let candidate = contacts[mid];
        if (candidate.a < key_hi || (candidate.a == key_hi && candidate.b < key_lo)) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo < count && contacts[lo].a == key_hi && contacts[lo].b == key_lo;
}

fn release(index: u32) {
    resting_live[index] = 0u;
    resting_next[index] = atomicExchange(&resting_free[0], index);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= min(atomicLoad(&resting_count[0]), arrayLength(&resting_live))) {
        return;
    }
    if (resting_live[index] == 0u) {
        return;
    }
    let contact = resting[index];
    let first_row = resolve_row(contact.first_body_id, contact.first_generation);
    let second_row = resolve_row(contact.second_body_id, contact.second_generation);
    if (first_row == NO_BODY || second_row == NO_BODY) {
        release(index);
        announce(COLLIDER_EVENT_BEGIN_END, EVENT_END, contact);
        return;
    }
    if (body_activity[first_row] == 0u && body_activity[second_row] == 0u) {
        return;
    }
    release(index);
    let key = contact_row_key(contact, first_row, second_row);
    if (current_holds(key.x, key.y)) {
        return;
    }
    announce(COLLIDER_EVENT_BEGIN_END, EVENT_END, contact);
}
