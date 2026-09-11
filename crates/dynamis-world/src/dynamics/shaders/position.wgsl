@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read> segments: array<u32>;
@group(0) @binding(5) var<storage, read> a_payload: array<u32>;
@group(0) @binding(6) var<storage, read_write> block_corrections: array<vec4f>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn store_block_correction(slot: u32, first: vec3f, second: vec3f) {
    block_corrections[slot * 2u] = vec4f(first, 0.0);
    block_corrections[slot * 2u + 1u] = vec4f(second, 0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = segments[SOLVER_BLOCK_CONTACT] + segments[SOLVER_BLOCK_CONSTRAINT];
    let stride = grid_stride(groups);
    for (var slot = global_index(gid); slot < live; slot = slot + stride) {
        let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
        let block = a_payload[slot];
        if (block < contact_blocks) {
            solve_contact_correction(block, slot);
        } else {
            store_block_correction(slot, vec3f(0.0), vec3f(0.0));
        }
    }
}
