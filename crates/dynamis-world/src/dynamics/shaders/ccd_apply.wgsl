@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> ccd_factor: array<u32>;
@group(0) @binding(3) var<storage, read> ccd_impact: array<vec4f>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.dynamic_count) {
        return;
    }
    let time = bitcast<f32>(ccd_factor[index]);
    if (time >= 1.0) {
        return;
    }
    let axis = ccd_impact[index].xyz;
    let restitution = ccd_impact[index].w;
    var state = body_states[index];
    state.position = state.prev_position + (state.position - state.prev_position) * time;
    let normal_speed = dot(state.velocity, axis);
    if (normal_speed > 0.0) {
        state.velocity = state.velocity - axis * normal_speed * (1.0 + restitution);
    }
    body_states[index] = state;
}
