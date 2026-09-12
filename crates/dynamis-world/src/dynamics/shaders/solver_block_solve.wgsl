@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(5) var<storage, read_write> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(6) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> segments: array<u32>;
@group(0) @binding(8) var<storage, read> a_payload: array<u32>;
@group(0) @binding(9) var<storage, read_write> block_deltas: array<vec4f>;
@group(0) @binding(10) var<storage, read> block_counts: array<u32>;
@group(0) @binding(11) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(12) var<storage, read> block_count: array<u32>;

struct BlockPair {
    first: Body,
    second: Body,
    split_first: Body,
    split_second: Body,
}

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn body_split(body: Body, scale: f32) -> Body {
    var split = body;
    split.desc.inverse_mass = body.desc.inverse_mass * scale;
    var inertia = body.desc.inverse_inertia;
    for (var index = 0u; index < 6u; index = index + 1u) {
        inertia[index] = inertia[index] * scale;
    }
    split.desc.inverse_inertia = inertia;
    return split;
}

fn block_scale(slot: u32) -> f32 {
    return f32(max(block_counts[slot], 1u));
}

fn block_bodies(first_slot: u32, second_slot: u32) -> BlockPair {
    var pair: BlockPair;
    pair.first = load_body(first_slot);
    pair.second = load_body(second_slot);
    pair.split_first = body_split(pair.first, block_scale(first_slot));
    pair.split_second = body_split(pair.second, block_scale(second_slot));
    if (body_is_inert(pair.first)) {
        pair.first = body_frozen(pair.first);
        pair.split_first = body_frozen(pair.split_first);
    }
    if (body_is_inert(pair.second)) {
        pair.second = body_frozen(pair.second);
        pair.split_second = body_frozen(pair.split_second);
    }
    return pair;
}

fn commit_block(
    slot: u32,
    first_slot: u32,
    second_slot: u32,
    delta_a: vec3f,
    spin_a: vec3f,
    delta_b: vec3f,
    spin_b: vec3f,
) {
    block_deltas[slot * 4u] = vec4f(delta_a, 0.0);
    block_deltas[slot * 4u + 1u] = vec4f(spin_a, 0.0);
    block_deltas[slot * 4u + 2u] = vec4f(delta_b, 0.0);
    block_deltas[slot * 4u + 3u] = vec4f(spin_b, 0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = block_count[0];
    let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
    let stride = grid_stride(groups);
    for (var slot = global_index(gid); slot < live; slot = slot + stride) {
        let block = a_payload[slot];
        if (block < contact_blocks) {
            solve_contact_block(block, slot);
        } else {
            solve_constraint_block(block - contact_blocks, slot);
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn warm(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = block_count[0];
    let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
    let stride = grid_stride(groups);
    for (var slot = global_index(gid); slot < live; slot = slot + stride) {
        let block = a_payload[slot];
        if (block < contact_blocks) {
            warm_contact_block(block, slot);
        } else {
            commit_block(slot, 0u, 0u, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
        }
    }
}
