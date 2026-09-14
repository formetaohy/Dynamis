@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(5) var<storage, read> segments: array<u32>;
@group(0) @binding(6) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(7) var<storage, read> constraint_rows: array<ConstraintRows>;
@group(0) @binding(8) var<storage, read_write> block_counts: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> target_speeds: array<f32>;
@group(0) @binding(10) var<storage, read_write> blocks: array<u32>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn extent() -> u32 {
    return segments[SOLVER_BLOCK_CONTACT] + segments[SOLVER_BLOCK_CONSTRAINT];
}

fn work(index: u32) {
    let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
    var first_body = NO_BODY;
    var second_body = NO_BODY;
    var resolves = false;
    if (index < contact_blocks) {
        let contact = contacts[index];
        first_body = collider_owners[contact.a];
        second_body = collider_owners[contact.b];
        if (contact_block_resolves(contact)) {
            resolves = true;
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
        }
    } else {
        let rows = constraint_rows[index - contact_blocks];
        first_body = rows.first_row;
        second_body = rows.second_row;
        resolves = constraint_runtime[index - contact_blocks].broken == 0u;
    }
    blocks[index * 2u] = first_body;
    blocks[index * 2u + 1u] = second_body;
    if (!resolves) {
        return;
    }
    atomicAdd(&block_counts[first_body], 1u);
    atomicAdd(&block_counts[second_body], 1u);
}
