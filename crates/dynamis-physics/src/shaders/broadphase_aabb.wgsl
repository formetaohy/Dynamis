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

struct Aabb {
    min: vec3f,
    _pad0: f32,
    max: vec3f,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read_write> aabbs: array<Aabb>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.body_count) {
        return;
    }
    let body = bodies[index];
    let radius = vec3f(body.radius);
    aabbs[index] = Aabb(body.position - radius, 0.0, body.position + radius, 0.0);
}
