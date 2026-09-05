@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.body_count) {
        return;
    }
    var body = bodies[index];
    if (body.inverse_mass == 0.0) {
        body.velocity = vec3f(0.0);
        body.angular_velocity = vec3f(0.0);
        body.force = vec3f(0.0);
        body.torque = vec3f(0.0);
        bodies[index] = body;
        return;
    }
    let linear_damping = 1.0 / (1.0 + params.damping * params.dt);
    let angular_damping = 1.0 / (1.0 + params.angular_damping * params.dt);
    body.velocity =
        (body.velocity + (params.gravity.xyz + body.force * body.inverse_mass) * params.dt) *
        linear_damping;
    body.angular_velocity =
        (body.angular_velocity + inverse_inertia(body) * body.torque * params.dt) *
        angular_damping;
    body.position = body.position + body.velocity * params.dt;
    let spin = vec4f(body.angular_velocity, 0.0);
    body.orientation =
        normalize(body.orientation + 0.5 * params.dt * quat_mul(spin, body.orientation));
    body.force = vec3f(0.0);
    body.torque = vec3f(0.0);
    bodies[index] = body;
}
