@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> contacts: array<Contact>;
@group(0) @binding(4) var<storage, read> segments: array<u32>;
@group(0) @binding(5) var<storage, read_write> resolution: array<vec4f>;
@group(0) @binding(6) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(7) var<storage, read> class_blocks: array<u32>;
@group(0) @binding(8) var<storage, read> class_range: array<u32>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn advance_position(row: u32, delta: vec3f) {
    var state = body_states[row];
    state.position = state.position + delta;
    body_states[row] = state;
    resolution[row] = vec4f(resolution[row].xyz + delta, 0.0);
}

fn gauss_seidel_contact_correction(contact_index: u32) {
    let contact = contacts[contact_index];
    if (!contact_block_resolves(contact)) {
        return;
    }
    let first_row = collider_owners[contact.a];
    let second_row = collider_owners[contact.b];
    let first_loaded = load_body(first_row);
    let second_loaded = load_body(second_row);
    var first = first_loaded;
    var second = second_loaded;
    let first_inert = body_is_inert(first_loaded);
    let second_inert = body_is_inert(second_loaded);
    if (first_inert) {
        first = body_frozen(first_loaded);
    }
    if (second_inert) {
        second = body_frozen(second_loaded);
    }
    let normal = contact.normal;
    let resolved = dot(normal, resolution[second_row].xyz - resolution[first_row].xyz);
    var total = vec3f(0.0);
    var contributing = 0u;
    for (var point_index = 0u; point_index < contact.point_count; point_index = point_index + 1u) {
        let error = max(contact.points[point_index].depth - params.slop - resolved, 0.0);
        if (error <= 0.0) {
            continue;
        }
        let k = linear_momentum_mass(first, second);
        if (k == 0.0) {
            continue;
        }
        total = total + normal * (params.relaxation * error / k);
        contributing = contributing + 1u;
    }
    if (contributing == 0u) {
        return;
    }
    let correction = total / f32(contributing);
    if (!first_inert) {
        advance_position(first_row, -correction * first.desc.inverse_mass);
    }
    if (!second_inert) {
        advance_position(second_row, correction * second.desc.inverse_mass);
    }
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
            gauss_seidel_contact_correction(block);
        }
    }
}
