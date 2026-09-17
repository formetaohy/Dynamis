@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(5) var<storage, read_write> block_counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> target_speeds: array<f32>;
@group(0) @binding(7) var<storage, read_write> blocks: array<u32>;
@group(0) @binding(8) var<storage, read_write> solver_rows: array<u32>;
@group(0) @binding(9) var<storage, read_write> solver_row_count: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> block_count: array<atomic<u32>>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn claim(row: u32) {
    if (atomicAdd(&block_counts[row], 1u) != 0u) {
        return;
    }
    solver_rows[atomicAdd(&solver_row_count[0], 1u)] = row;
}

fn work(index: u32) {
    let contact = contacts[index];
    let first_body = collider_owners[contact.a];
    let second_body = collider_owners[contact.b];
    blocks[index * 2u] = first_body;
    blocks[index * 2u + 1u] = second_body;
    if (!contact_block_resolves(contact)) {
        return;
    }
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
        target_speeds[index * CONTACT_MAX_POINTS + point_index] = target_speed;
    }
    claim(first_body);
    claim(second_body);
}
