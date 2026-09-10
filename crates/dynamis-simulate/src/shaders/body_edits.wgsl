struct BodyCommand {
    kind: u32,
    slot: u32,
    mask: u32,
    _pad0: u32,
    state: BodyState,
}

@group(0) @binding(0) var<storage, read> commands: array<BodyCommand>;
@group(0) @binding(1) var<storage, read> command_first: array<u32>;
@group(0) @binding(2) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(3) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(4) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: SimParams;

fn wake(body: ptr<function, BodyState>, slot: u32) {
    if ((*body).sleeping != 0u) {
        atomicOr(&wake_flags[slot], 1u);
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
    let body_index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (body_index >= params.body_count) {
        return;
    }
    let first_entry = command_first[body_index];
    if (first_entry == 0u) {
        return;
    }
    var cursor = first_entry - 1u;
    loop {
        if (cursor >= arrayLength(&commands)) {
            break;
        }
        let command = commands[cursor];
        if (command.slot != body_index) {
            break;
        }
        var state = body_states[body_index];
        let desc = body_descs[body_index];
        if (command.kind == COMMAND_PATCH) {
            if ((command.mask & PATCH_POSITION) != 0u) {
                state.position = command.state.position;
                state.prev_position = command.state.position;
            }
            if ((command.mask & PATCH_ORIENTATION) != 0u) {
                state.orientation = command.state.orientation;
            }
            if ((command.mask & PATCH_VELOCITY) != 0u) {
                state.velocity = command.state.velocity;
            }
            if ((command.mask & PATCH_ANGULAR_VELOCITY) != 0u) {
                state.angular_velocity = command.state.angular_velocity;
            }
            wake(&state, body_index);
        } else if (command.kind == COMMAND_FORCE) {
            state.force = state.force + command.state.force;
            wake(&state, body_index);
        } else if (command.kind == COMMAND_FORCE_AT_POINT) {
            state.force = state.force + command.state.force;
            state.torque = state.torque
                + cross(command.state.position - body_com_of(state, desc), command.state.force);
            wake(&state, body_index);
        } else if (command.kind == COMMAND_TORQUE) {
            state.torque = state.torque + command.state.torque;
            wake(&state, body_index);
        } else if (command.kind == COMMAND_IMPULSE) {
            state.velocity = state.velocity + command.state.velocity * desc.inverse_mass;
            wake(&state, body_index);
        } else if (command.kind == COMMAND_IMPULSE_AT_POINT) {
            apply_impulse_at(&state, desc, command.state.velocity, command.state.position);
            wake(&state, body_index);
        } else if (command.kind == COMMAND_ANGULAR_IMPULSE) {
            state.angular_velocity = state.angular_velocity
                + apply_inverse_inertia_of(desc, state.orientation, command.state.angular_velocity);
            wake(&state, body_index);
        } else if (command.kind == COMMAND_SLEEP) {
            state.velocity = vec3f(0.0);
            state.angular_velocity = vec3f(0.0);
            state.sleep_timer = 0.0;
            state.sleeping = 1u;
            atomicStore(&wake_flags[body_index], 0u);
        } else if (command.kind == COMMAND_WAKE) {
            state.sleep_timer = 0.0;
            state.sleeping = 0u;
            atomicOr(&wake_flags[body_index], 1u);
        }
        body_states[body_index] = state;
        cursor = cursor + 1u;
    }
}
