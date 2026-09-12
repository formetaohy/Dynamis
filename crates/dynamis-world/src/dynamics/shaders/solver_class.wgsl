@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(5) var<storage, read_write> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(6) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> segments: array<u32>;
@group(0) @binding(8) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(9) var<storage, read> class_blocks: array<u32>;
@group(0) @binding(10) var<storage, read> class_range: array<u32>;

struct BlockPair {
    first: Body,
    second: Body,
    split_first: Body,
    split_second: Body,
}

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn block_bodies(first_slot: u32, second_slot: u32) -> BlockPair {
    var pair: BlockPair;
    pair.first = load_body(first_slot);
    pair.second = load_body(second_slot);
    pair.split_first = pair.first;
    pair.split_second = pair.second;
    if (body_is_inert(pair.first)) {
        pair.first = body_frozen(pair.first);
        pair.split_first = body_frozen(pair.split_first);
    }
    if (body_is_inert(pair.second)) {
        pair.second = body_frozen(pair.second);
        pair.split_second = body_frozen(pair.second);
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
    apply_body_delta(first_slot, delta_a, spin_a);
    apply_body_delta(second_slot, delta_b, spin_b);
}

fn apply_body_delta(slot: u32, delta: vec3f, spin: vec3f) {
    var body = load_body(slot);
    if (!body_is_movable(body.desc) || body.state.sleeping != 0u) {
        return;
    }
    body.state.velocity = body.state.velocity + delta;
    body.state.angular_velocity = body.state.angular_velocity + spin;
    body_states[slot] = body.state;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let start = class_range[0];
    let end = class_range[1];
    let contact_blocks = segments[SOLVER_BLOCK_CONTACT];
    let stride = grid_stride(groups);
    for (var slot = start + global_index(gid); slot < end; slot = slot + stride) {
        let block = class_blocks[slot];
        if (block < contact_blocks) {
            solve_contact_block(block, slot);
        } else {
            solve_constraint_block(block - contact_blocks, slot);
        }
    }
}
