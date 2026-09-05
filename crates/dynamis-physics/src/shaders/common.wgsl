struct SimParams {
    gravity: vec4f,
    dt: f32,
    damping: f32,
    angular_damping: f32,
    body_count: u32,
    relaxation: f32,
    slop: f32,
    restitution_threshold: f32,
}

struct RigidBody {
    position: vec3f,
    _pad0: f32,
    orientation: vec4f,
    velocity: vec3f,
    _pad1: f32,
    angular_velocity: vec3f,
    _pad2: f32,
    inverse_mass: f32,
    radius: f32,
    restitution: f32,
    friction: f32,
    force: vec3f,
    _pad3: f32,
    torque: vec3f,
    _pad4: f32,
}

struct Aabb {
    min: vec3f,
    _pad0: f32,
    max: vec3f,
    _pad1: f32,
}

struct Pair {
    a: u32,
    b: u32,
}

struct Contact {
    a: u32,
    b: u32,
    depth: f32,
    _pad: f32,
    normal: vec3f,
}

fn inverse_inertia(body: RigidBody) -> f32 {
    return 2.5 * body.inverse_mass / (body.radius * body.radius);
}

fn quat_mul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(
        a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz),
    );
}
