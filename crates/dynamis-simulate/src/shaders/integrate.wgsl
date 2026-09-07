@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.body_count) {
        return;
    }
    var body = bodies[index];
    let kinematic = (body.flags & BODY_KINEMATIC) != 0u;
    if ((body.flags & BODY_SLEEPING) != 0u) {
        body.velocity = vec3f(0.0);
        body.angular_velocity = vec3f(0.0);
        body.force = vec3f(0.0);
        body.torque = vec3f(0.0);
        bodies[index] = body;
        return;
    }
    if (body.inverse_mass == 0.0 && !kinematic) {
        body.prev_position = body.position;
        body.velocity = vec3f(0.0);
        body.angular_velocity = vec3f(0.0);
        body.force = vec3f(0.0);
        body.torque = vec3f(0.0);
        bodies[index] = body;
        return;
    }
    let q = body.orientation;
    let linear_damping = 1.0 / (1.0 + params.damping * params.dt);
    let angular_damping = 1.0 / (1.0 + params.angular_damping * params.dt);
    if (!kinematic) {
        body.velocity =
            body.velocity + (params.gravity.xyz + body.force * body.inverse_mass) * params.dt;
        body.angular_velocity =
            body.angular_velocity + apply_inverse_inertia(body, body.torque * params.dt);
    }
    let speed = length(body.velocity);
    if (speed > params.max_velocity) {
        body.velocity = body.velocity * (params.max_velocity / speed);
    }
    let spin = length(body.angular_velocity);
    if (spin > params.max_angular_velocity) {
        body.angular_velocity = body.angular_velocity * (params.max_angular_velocity / spin);
    }
    body.velocity = body.velocity * linear_damping;
    body.angular_velocity = body.angular_velocity * angular_damping;
    body.prev_position = body.position;
    body.position = body.position + body.velocity * params.dt;
    let spin_quat = vec4f(body.angular_velocity, 0.0);
    body.orientation = normalize(body.orientation + 0.5 * params.dt * quat_mul(spin_quat, body.orientation));
    body.force = vec3f(0.0);
    body.torque = vec3f(0.0);
    bodies[index] = body;
}
