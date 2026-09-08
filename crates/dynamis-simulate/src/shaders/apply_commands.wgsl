struct BodyCommand {
    kind: u32,
    slot: u32,
    mask: u32,
    _pad0: u32,
    state: BodyState,
}

@group(0) @binding(0) var<storage, read> commands: array<BodyCommand>;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> command_count: u32;
@group(0) @binding(4) var<storage, read_write> wake_flags: array<atomic<u32>>;

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
    if (gid.x != 0u) {
        return;
    }
    let count = command_count;
    for (var i = 0u; i < count; i = i + 1u) {
        let command = commands[i];
        let slot = command.slot;
        if (command.kind == COMMAND_ADD) {
            body_states[slot] = command.state;
            continue;
        }
        if (command.kind == COMMAND_REMOVE) {
            body_states[slot] = body_states[command.mask];
            body_states[command.mask] = BodyState();
            continue;
        }
        if (command.kind == COMMAND_SWAP) {
            let first = body_states[slot];
            let second = body_states[command.mask];
            body_states[slot] = second;
            body_states[command.mask] = first;
            continue;
        }
        var state = body_states[slot];
        let desc = body_descs[slot];
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
            wake(&state, slot);
        } else if (command.kind == COMMAND_FORCE) {
            state.force = state.force + command.state.force;
            wake(&state, slot);
        } else if (command.kind == COMMAND_FORCE_AT_POINT) {
            state.force = state.force + command.state.force;
            state.torque = state.torque
                + cross(command.state.position - body_com_of(state, desc), command.state.force);
            wake(&state, slot);
        } else if (command.kind == COMMAND_TORQUE) {
            state.torque = state.torque + command.state.torque;
            wake(&state, slot);
        } else if (command.kind == COMMAND_IMPULSE) {
            state.velocity = state.velocity + command.state.velocity * desc.inverse_mass;
            wake(&state, slot);
        } else if (command.kind == COMMAND_IMPULSE_AT_POINT) {
            apply_impulse_at(&state, desc, command.state.velocity, command.state.position);
            wake(&state, slot);
        } else if (command.kind == COMMAND_ANGULAR_IMPULSE) {
            state.angular_velocity = state.angular_velocity
                + apply_inverse_inertia_of(desc, state.orientation, command.state.angular_velocity);
            wake(&state, slot);
        } else if (command.kind == COMMAND_SLEEP) {
            state.velocity = vec3f(0.0);
            state.angular_velocity = vec3f(0.0);
            state.sleep_timer = 0.0;
            state.sleeping = 1u;
            atomicStore(&wake_flags[slot], 0u);
        } else if (command.kind == COMMAND_WAKE) {
            state.sleep_timer = 0.0;
            state.sleeping = 0u;
            atomicOr(&wake_flags[slot], 1u);
        }
        body_states[slot] = state;
    }
}
