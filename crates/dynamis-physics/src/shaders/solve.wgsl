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

struct Contact {
    a: u32,
    b: u32,
    depth: f32,
    _pad: f32,
    normal: vec3f,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read> contacts: array<Contact>;
@group(0) @binding(3) var<storage, read> contact_count: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= atomicLoad(&contact_count)) {
        return;
    }
    let contact = contacts[index];
    var first = bodies[contact.a];
    var second = bodies[contact.b];
    let first_weight = first.inverse_mass;
    let second_weight = second.inverse_mass;
    let weight_sum = first_weight + second_weight;
    if (weight_sum == 0.0) {
        return;
    }
    let delta = second.position - first.position;
    let distance = length(delta);
    let radius_sum = first.radius + second.radius;
    let depth = radius_sum - distance;
    if (depth <= 0.0) {
        return;
    }
    var normal = vec3f(0.0, 1.0, 0.0);
    if (distance > 1e-6) {
        normal = delta / distance;
    }
    let correction = depth * params.relaxation / weight_sum;
    first.position = first.position - normal * (correction * first_weight);
    second.position = second.position + normal * (correction * second_weight);
    let relative_velocity = dot(second.velocity - first.velocity, normal);
    if (relative_velocity < 0.0) {
        let restitution = max(first.restitution, second.restitution);
        let impulse = -(1.0 + restitution) * relative_velocity / weight_sum;
        first.velocity = first.velocity - normal * (impulse * first_weight);
        second.velocity = second.velocity + normal * (impulse * second_weight);
    }
    bodies[contact.a] = first;
    bodies[contact.b] = second;
}
