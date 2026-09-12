@group(0) @binding(0) var<storage, read> archive: array<Contact>;
@group(0) @binding(1) var<storage, read_write> archive_count: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(3) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> contact_matched: array<u32>;
@group(0) @binding(5) var<storage, read_write> events: array<ContactEvent>;
@group(0) @binding(6) var<storage, read_write> event_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> spillover: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params: StepParams;
@group(0) @binding(9) var<storage, read> row_of_body: array<u32>;
@group(0) @binding(10) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(11) var<storage, read> body_descs: array<BodyDescriptor>;

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

fn current_slot(contact: Contact) -> u32 {
    let count = min(atomicLoad(&contact_count[0]), arrayLength(&contacts));
    var lo = 0u;
    var hi = count;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let candidate = contacts[mid];
        if (candidate.a < contact.a || (candidate.a == contact.a && candidate.b < contact.b)) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo < count && contacts[lo].a == contact.a && contacts[lo].b == contact.b) {
        return lo;
    }
    return NO_SLOT;
}

fn extent() -> u32 {
    return min(atomicLoad(&archive_count[0]), arrayLength(&archive));
}

fn work(index: u32) {
    let held = archive[index];
    let first_row = resolve_row(held.first_body_id, held.first_generation);
    let second_row = resolve_row(held.second_body_id, held.second_generation);
    if (first_row != NO_BODY && second_row != NO_BODY) {
        let slot = current_slot(held);
        if (slot != NO_SLOT) {
            let current = contacts[slot];
            if (contact_same_pair(held, current)) {
                if ((held.events & CONTACT_ANNOUNCED) != 0u) {
                    contact_matched[slot] = 1u;
                    var relayed = current;
                    relayed.events = relayed.events | CONTACT_ANNOUNCED;
                    if (contact_carries_over(held, current)) {
                        if (contact_touches(current, params.slop)) {
                            announce(COLLIDER_EVENT_PERSIST, EVENT_PERSIST, current);
                        }
                        relayed = contact_relay_impulses(current, held);
                        relayed.events = relayed.events | CONTACT_ANNOUNCED;
                    }
                    contacts[slot] = relayed;
                }
                return;
            }
        }
    }
    if ((held.events & COLLIDER_EVENT_BEGIN_END) == 0u) {
        return;
    }
    if (first_row != NO_BODY && second_row != NO_BODY
    && !body_is_active(body_states[first_row], body_descs[first_row])
    && !body_is_active(body_states[second_row], body_descs[second_row])) {
        return;
    }
    if ((held.events & CONTACT_ANNOUNCED) != 0u) {
        announce(COLLIDER_EVENT_BEGIN_END, EVENT_END, held);
    }
}

