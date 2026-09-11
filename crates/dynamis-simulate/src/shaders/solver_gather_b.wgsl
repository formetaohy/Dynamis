@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(2) var<storage, read> segments: array<u32>;
@group(0) @binding(3) var<storage, read> a_payload: array<u32>;
@group(0) @binding(4) var<storage, read_write> b_bodies: array<u32>;
@group(0) @binding(5) var<storage, read_write> b_blocks: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let slot = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
    if (slot >= contact_blocks + segments[SOLVER_BLOCK_CONSTRAINT]) {
        return;
    }
    let block = a_payload[slot];
    var body_b = 0u;
    if (block < contact_blocks) {
        body_b = contacts[block].b / MAX_COLLIDERS_PER_BODY;
    } else {
        body_b = constraint_descs[block - contact_blocks].b;
    }
    b_bodies[slot] = body_b;
    b_blocks[slot] = slot;
}
