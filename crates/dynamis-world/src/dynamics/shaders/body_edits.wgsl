struct BodyEdit {
    kind: u32,
    mask: u32,
    _pad0: u32,
    _pad1: u32,
    state: BodyState,
}

struct BodyEditRun {
    row: u32,
    first: u32,
    len: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> edits: array<BodyEdit>;
@group(0) @binding(1) var<storage, read> edit_runs: array<BodyEditRun>;
@group(0) @binding(2) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(3) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(4) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: StepParams;
@group(0) @binding(6) var<storage, read_write> slept_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> woke_count: array<atomic<u32>>;

fn mark_awake(body: ptr<function, BodyState>, row: u32) {
    if ((*body).sleeping != 0u) {
        atomicOr(&wake_flags[row], 1u);
        atomicAdd(&woke_count[0], 1u);
    }
    (*body).sleeping = 0u;
    (*body).sleep_timer = 0.0;
}

fn apply_impulse_at(
    state: ptr<function, BodyState>,
    desc: BodyDescriptor,
    impulse: vec3f,
    point: vec3f,
) {
    (*state).velocity = (*state).velocity + impulse * desc.inverse_mass;
    let lever = point - body_com_of(*state, desc);
    (*state).angular_velocity = (*state).angular_velocity
        + apply_inverse_inertia_of(desc, (*state).orientation, cross(lever, impulse));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= min(params.edit_run_count, arrayLength(&edit_runs))) {
        return;
    }
    let run = edit_runs[index];
    if (run.len == 0u) {
        return;
    }
    var state = body_states[run.row];
    let desc = body_descs[run.row];
    for (var i = 0u; i < run.len; i = i + 1u) {
        let edit = edits[run.first + i];
        if (edit.kind == EDIT_PATCH) {
            if ((edit.mask & PATCH_POSITION) != 0u) {
                state.position = edit.state.position;
                state.prev_position = edit.state.position;
            }
            if ((edit.mask & PATCH_ORIENTATION) != 0u) {
                state.orientation = edit.state.orientation;
            }
            if ((edit.mask & PATCH_VELOCITY) != 0u) {
                state.velocity = edit.state.velocity;
            }
            if ((edit.mask & PATCH_ANGULAR_VELOCITY) != 0u) {
                state.angular_velocity = edit.state.angular_velocity;
            }
            mark_awake(&state, run.row);
        } else if (edit.kind == EDIT_FORCE) {
            state.force = state.force + edit.state.force;
            mark_awake(&state, run.row);
        } else if (edit.kind == EDIT_FORCE_AT_POINT) {
            state.force = state.force + edit.state.force;
            state.torque = state.torque
                + cross(edit.state.position - body_com_of(state, desc), edit.state.force);
            mark_awake(&state, run.row);
        } else if (edit.kind == EDIT_TORQUE) {
            state.torque = state.torque + edit.state.torque;
            mark_awake(&state, run.row);
        } else if (edit.kind == EDIT_IMPULSE) {
            state.velocity = state.velocity + edit.state.velocity * desc.inverse_mass;
            mark_awake(&state, run.row);
        } else if (edit.kind == EDIT_IMPULSE_AT_POINT) {
            apply_impulse_at(&state, desc, edit.state.velocity, edit.state.position);
            mark_awake(&state, run.row);
        } else if (edit.kind == EDIT_ANGULAR_IMPULSE) {
            state.angular_velocity = state.angular_velocity
                + apply_inverse_inertia_of(desc, state.orientation, edit.state.angular_velocity);
            mark_awake(&state, run.row);
        } else if (edit.kind == EDIT_SLEEP) {
            if (state.sleeping == 0u) {
                atomicAdd(&slept_count[0], 1u);
            }
            state.velocity = vec3f(0.0);
            state.angular_velocity = vec3f(0.0);
            state.sleep_timer = 0.0;
            state.sleeping = 1u;
            atomicStore(&wake_flags[run.row], 0u);
        } else if (edit.kind == EDIT_WAKE) {
            if (state.sleeping != 0u) {
                atomicAdd(&woke_count[0], 1u);
            }
            state.sleep_timer = 0.0;
            state.sleeping = 0u;
            atomicOr(&wake_flags[run.row], 1u);
        }
    }
    body_states[run.row] = state;
}
