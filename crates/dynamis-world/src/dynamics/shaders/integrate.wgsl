@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;

fn work(index: u32) {
    var state = body_states[index];
    let desc = body_descs[index];
    if (state.sleeping != 0u) {
        state.velocity = vec3f(0.0);
        state.angular_velocity = vec3f(0.0);
        state.force = vec3f(0.0);
        state.torque = vec3f(0.0);
        body_states[index] = state;
        return;
    }
    let kinematic = (desc.flags & BODY_KINEMATIC) != 0u;
    if (desc.inverse_mass == 0.0 && !kinematic) {
        state.prev_position = state.position;
        state.velocity = vec3f(0.0);
        state.angular_velocity = vec3f(0.0);
        state.force = vec3f(0.0);
        state.torque = vec3f(0.0);
        body_states[index] = state;
        return;
    }
    let q = state.orientation;
    let linear_damping = 1.0 / (1.0 + desc.linear_damping * params.dt);
    let angular_damping = 1.0 / (1.0 + desc.angular_damping * params.dt);
    if (!kinematic) {
        state.velocity =
            state.velocity + (params.gravity.xyz * desc.gravity_scale + state.force * desc.inverse_mass) * params.dt;
        state.angular_velocity =
            state.angular_velocity + apply_inverse_inertia_of(desc, q, state.torque * params.dt);
    }
    let speed = length(state.velocity);
    if (speed > params.max_velocity) {
        state.velocity = state.velocity * (params.max_velocity / speed);
    }
    let spin = length(state.angular_velocity);
    if (spin > params.max_angular_velocity) {
        state.angular_velocity = state.angular_velocity * (params.max_angular_velocity / spin);
    }
    state.velocity = state.velocity * linear_damping;
    state.angular_velocity = state.angular_velocity * angular_damping;
    state.prev_position = state.position;
    state.force = vec3f(0.0);
    state.torque = vec3f(0.0);
    body_states[index] = state;
}
