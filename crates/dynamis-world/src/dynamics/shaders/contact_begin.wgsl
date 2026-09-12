@group(0) @binding(0) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> contact_matched: array<u32>;
@group(0) @binding(3) var<storage, read_write> events: array<ContactEvent>;
@group(0) @binding(4) var<storage, read_write> event_count: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> spillover: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: StepParams;
@group(0) @binding(7) var<storage, read> resting: array<Contact>;
@group(0) @binding(8) var<storage, read> resting_live: array<u32>;
@group(0) @binding(9) var<storage, read> resting_major: array<u32>;
@group(0) @binding(10) var<storage, read> resting_minor: array<u32>;
@group(0) @binding(11) var<storage, read> resting_slots: array<u32>;
@group(0) @binding(12) var<storage, read_write> resting_index: array<atomic<u32>>;

fn resting_slot(contact: Contact) -> u32 {
    let count = min(atomicLoad(&resting_index[0]), arrayLength(&resting_slots));
    let key = contact_pair_key(contact);
    var lo = 0u;
    var hi = count;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let major = resting_major[mid];
        if (major < key.x || (major == key.x && resting_minor[mid] < key.y)) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    while (lo < count && resting_major[lo] == key.x && resting_minor[lo] == key.y) {
        let slot = resting_slots[lo];
        if (slot < arrayLength(&resting_live) && resting_live[slot] == 1u
            && contact_same_pair(resting[slot], contact)) {
            return slot;
        }
        lo = lo + 1u;
    }
    return NO_SLOT;
}

fn extent() -> u32 {
    return min(atomicLoad(&contact_count[0]), arrayLength(&contacts));
}

fn work(index: u32) {
    if (contact_matched[index] != 0u) {
        return;
    }
    let contact = contacts[index];
    let slot = resting_slot(contact);
    if (slot != NO_SLOT) {
        var revived = contact;
        if ((resting[slot].events & CONTACT_ANNOUNCED) != 0u) {
            revived.events = revived.events | CONTACT_ANNOUNCED;
            if (contact_touches(contact, params.slop)) {
                announce(COLLIDER_EVENT_PERSIST, EVENT_PERSIST, contact);
            }
            if (contact_carries_over(resting[slot], contact)) {
                revived = contact_relay_impulses(revived, resting[slot]);
                revived.events = revived.events | CONTACT_ANNOUNCED;
            }
        } else if (contact_touches(contact, params.slop)) {
            announce(COLLIDER_EVENT_BEGIN_END, EVENT_BEGIN, contact);
            revived.events = revived.events | CONTACT_ANNOUNCED;
        }
        contacts[index] = revived;
        return;
    }
    if (contact_touches(contact, params.slop)) {
        announce(COLLIDER_EVENT_BEGIN_END, EVENT_BEGIN, contact);
        var announced = contacts[index];
        announced.events = announced.events | CONTACT_ANNOUNCED;
        contacts[index] = announced;
    }
}

