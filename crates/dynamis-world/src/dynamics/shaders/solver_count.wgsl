@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(5) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(6) var<storage, read> segments: array<u32>;
@group(0) @binding(7) var<storage, read_write> a_bodies: array<u32>;
@group(0) @binding(8) var<storage, read_write> a_payload: array<u32>;
@group(0) @binding(9) var<storage, read_write> block_counts: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> contact_counts: array<atomic<u32>>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = segments[SOLVER_BLOCK_CONTACT] + segments[SOLVER_BLOCK_CONSTRAINT];
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {
        let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
        a_payload[index] = index;
        if (index < contact_blocks) {
            var contact = contacts[index];
            let body_a = contact.a / MAX_COLLIDERS_PER_BODY;
            a_bodies[index] = body_a;
            if (contact_block_resolves(contact)) {
                let first = load_body(body_a);
                let second = load_body(contact.b / MAX_COLLIDERS_PER_BODY);
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
                let body_b = contact.b / MAX_COLLIDERS_PER_BODY;
                atomicAdd(&block_counts[body_a], 1u);
                atomicAdd(&block_counts[body_b], 1u);
                atomicAdd(&contact_counts[body_a], 1u);
                atomicAdd(&contact_counts[body_b], 1u);
            }
        } else {
            let constraint_index = index - contact_blocks;
            let constraint = constraint_descs[constraint_index];
            a_bodies[index] = constraint.a;
            if (constraint_runtime[constraint_index].broken == 0u) {
                atomicAdd(&block_counts[constraint.a], 1u);
                atomicAdd(&block_counts[constraint.b], 1u);
            }
        }
    }
}
