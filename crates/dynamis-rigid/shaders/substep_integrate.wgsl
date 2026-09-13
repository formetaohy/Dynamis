@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> live_bodies: array<u32>;
@group(0) @binding(4) var<storage, read> live_count: array<u32>;

const GYROSCOPIC_ITERATIONS: u32 = 4u;

fn extent() -> u32 {
    return min(live_count[0], arrayLength(&live_bodies));
}

fn gyroscopic_spin(desc: BodyDescriptor, q: vec4f, spin: vec3f) -> vec3f {
    if (inertia_is_isotropic(desc)) {
        return spin;
    }
    let body_spin = quat_rotate(quat_conjugate(q), spin);
    let momentum = inertia_local(desc, body_spin);
    if (all(cross(body_spin, momentum) == vec3f(0.0))) {
        return spin;
    }
    var solved = momentum;
    for (var iteration = 0u; iteration < GYROSCOPIC_ITERATIONS; iteration = iteration + 1u) {
        let midpoint = 0.5 * (momentum + solved);
        solved = rotate_about(-inverse_inertia_local(desc, midpoint) * params.substep_dt, momentum);
    }
    return quat_rotate(q, inverse_inertia_local(desc, solved));
}

fn work(index: u32) {
    let row = live_bodies[index];
    var state = body_states[row];
    let desc = body_descs[row];
    let q = state.orientation;
    let kinematic = (desc.flags & BODY_KINEMATIC) != 0u;
    if (!kinematic) {
        state.velocity =
            state.velocity
            + (params.gravity.xyz * desc.gravity_scale + state.force * desc.inverse_mass) * params.substep_dt;
        state.angular_velocity =
            gyroscopic_spin(desc, q, state.angular_velocity)
            + apply_inverse_inertia_of(desc, q, state.torque * params.substep_dt);
    }
    let speed = length(state.velocity);
    if (speed > params.max_velocity) {
        state.velocity = state.velocity * (params.max_velocity / speed);
    }
    let spin = length(state.angular_velocity);
    if (spin > params.max_angular_velocity) {
        state.angular_velocity = state.angular_velocity * (params.max_angular_velocity / spin);
    }
    state.velocity = state.velocity / (1.0 + desc.linear_damping * params.substep_dt);
    state.angular_velocity = state.angular_velocity / (1.0 + desc.angular_damping * params.substep_dt);
    body_states[row] = state;
}
