@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(5) var<storage, read> segments: array<u32>;
@group(0) @binding(6) var<storage, read_write> block_first_body: array<u32>;
@group(0) @binding(7) var<storage, read_write> block_second_body: array<u32>;
@group(0) @binding(8) var<storage, read_write> a_bodies: array<u32>;
@group(0) @binding(9) var<storage, read_write> a_payload: array<u32>;
@group(0) @binding(10) var<storage, read> collider_owners: array<u32>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn extent() -> u32 {
    return segments[SOLVER_BLOCK_CONTACT] + segments[SOLVER_BLOCK_CONSTRAINT];
}

fn work(index: u32) {
    let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
    a_payload[index] = index;
    if (index < contact_blocks) {
        var contact = contacts[index];
        let first_body = collider_owners[contact.a];
        let second_body = collider_owners[contact.b];
        block_first_body[index] = first_body;
        block_second_body[index] = second_body;
        a_bodies[index] = first_body;
        if (contact_block_resolves(contact)) {
            let first = load_body(first_body);
            let second = load_body(second_body);
            for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
                let point = contact.points[point_index];
                let speed = dot(relative_velocity(first, second, point.position, point.position), contact.normal);
                var target_speed = 0.0;
                if (point.depth < 0.0) {
                    target_speed = point.depth / params.dt;
                } else if (speed < -params.restitution_threshold) {
                    target_speed = -speed * contact.restitution;
                }
                contact.points[point_index].target_speed = target_speed;
            }
            contacts[index] = contact;
        }
    } else {
        let constraint = constraint_descs[index - contact_blocks];
        block_first_body[index] = constraint.a;
        block_second_body[index] = constraint.b;
        a_bodies[index] = constraint.a;
    }
}

