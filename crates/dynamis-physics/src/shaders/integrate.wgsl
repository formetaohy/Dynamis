struct SimParams {
    gravity: vec4f,
    dt: f32,
    damping: f32,
    body_count: u32,
    relaxation: f32,
}

struct RigidBody {
    position: vec3f,
    _pad0: f32,
    velocity: vec3f,
    _pad1: f32,
    inverse_mass: f32,
    radius: f32,
    restitution: f32,
    _pad2: f32,
}

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
        bodies[index] = body;
        return;
    }
    let damping_factor = 1.0 / (1.0 + params.damping * params.dt);
    body.velocity = (body.velocity + params.gravity.xyz * params.dt) * damping_factor;
    body.position = body.position + body.velocity * params.dt;
    bodies[index] = body;
}
