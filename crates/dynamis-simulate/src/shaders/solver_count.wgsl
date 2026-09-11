@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(2) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(3) var<storage, read> segments: array<u32>;
@group(0) @binding(4) var<storage, read_write> a_bodies: array<u32>;
@group(0) @binding(5) var<storage, read_write> a_payload: array<u32>;
@group(0) @binding(6) var<storage, read_write> block_counts: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> contact_counts: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
    if (index >= contact_blocks + segments[SOLVER_BLOCK_CONSTRAINT]) {
        return;
    }
    a_payload[index] = index;
    if (index < contact_blocks) {
        let contact = contacts[index];
        let body_a = contact.a / MAX_COLLIDERS_PER_BODY;
        a_bodies[index] = body_a;
        if (contact_block_resolves(contact)) {
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
