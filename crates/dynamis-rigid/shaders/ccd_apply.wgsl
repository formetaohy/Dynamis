@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> ccd_factor: array<u32>;
@group(0) @binding(3) var<storage, read> ccd_impact: array<CcdImpact>;
@group(0) @binding(4) var<storage, read> body_descs: array<BodyDescriptor>;

fn work(index: u32) {
    let time = bitcast<f32>(ccd_factor[index]);
    if (time >= 1.0) {
        return;
    }
    let impact = ccd_impact[index];
    let desc = body_descs[index];
    let axis = impact.normal;
    var state = body_states[index];
    let lever = impact.point - body_com_of(state, desc);
    let arm = cross(lever, axis);
    let linear_speed = dot(state.velocity, axis);
    let angular_speed = dot(state.angular_velocity, arm);
    if (linear_speed <= 0.0 && angular_speed <= 0.0) {
        return;
    }
    state.position = state.prev_position + (state.position - state.prev_position) * time;
    state.orientation = quat_slerp(state.prev_orientation, state.orientation, time);
    if (linear_speed > 0.0) {
        state.velocity = state.velocity - axis * (linear_speed * (1.0 + impact.restitution));
    }
    if (angular_speed > 0.0) {
        let mass = desc.inverse_mass + dot(arm, apply_inverse_inertia_of(desc, state.orientation, arm));
        if (mass > 0.0) {
            let impulse = angular_speed * (1.0 + impact.restitution) / mass;
            state.velocity = state.velocity - axis * (impulse * desc.inverse_mass);
            state.angular_velocity = state.angular_velocity
                - apply_inverse_inertia_of(desc, state.orientation, arm * impulse);
        }
    }
    body_states[index] = state;
}
