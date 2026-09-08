@group(0) @binding(0) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(1) var<storage, read> contacts: array<Contact>;
@group(0) @binding(2) var<storage, read_write> contact_count: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> island_parents: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> wake_flags: array<atomic<u32>>;

fn island_link(a: u32, b: u32) {
    atomicMin(&island_parents[a], b);
    atomicMin(&island_parents[b], a);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= atomicLoad(&contact_count)) {
        return;
    }
    let contact = contacts[index];
    if (contact.point_count == 0u || contact.sensor == 1u) {
        return;
    }
    let first = bodies[contact.a / MAX_COLLIDERS_PER_BODY];
    let second = bodies[contact.b / MAX_COLLIDERS_PER_BODY];
    let first_static = body_is_static(first);
    let second_static = body_is_static(second);
    if (first_static) {
        if (atomicLoad(&wake_flags[contact.a / MAX_COLLIDERS_PER_BODY]) != 0u) {
            atomicOr(&wake_flags[contact.b / MAX_COLLIDERS_PER_BODY], 1u);
        }
    }
    if (second_static) {
        if (atomicLoad(&wake_flags[contact.b / MAX_COLLIDERS_PER_BODY]) != 0u) {
            atomicOr(&wake_flags[contact.a / MAX_COLLIDERS_PER_BODY], 1u);
        }
    }
    if (!first_static && !second_static &&
        body_is_dynamic(first) && (first.flags & BODY_SLEEPING) == 0u &&
        body_is_dynamic(second) && (second.flags & BODY_SLEEPING) == 0u) {
        island_link(contact.a / MAX_COLLIDERS_PER_BODY, contact.b / MAX_COLLIDERS_PER_BODY);
    }
}
