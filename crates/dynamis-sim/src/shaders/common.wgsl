struct SimParams {
    gravity: vec4f,
    dt: f32,
    damping: f32,
    angular_damping: f32,
    body_count: u32,
    solve_iterations: u32,
    position_iterations: u32,
    constraint_count: u32,
    relaxation: f32,
    slop: f32,
    restitution_threshold: f32,
    max_velocity: f32,
    max_angular_velocity: f32,
    grid_cell_size: f32,
    max_cells_per_body: u32,
    _pad0: u32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
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
    restitution: f32,
    friction: f32,
    body_id: u32,
    generation: u32,
    _pad_collider: u32,
    flags: u32,
    collision_group: u32,
    collision_mask: u32,
    _pad6: u32,
    _pad7: u32,
    _pad8: u32,
    inverse_inertia_body: vec3f,
    _pad3: f32,
    force: vec3f,
    _pad4: f32,
    torque: vec3f,
    _pad5: f32,
}

struct Collider {
    shape: u32,
    _pad0: u32,
    radius: f32,
    half_height: f32,
    half_extents: vec3f,
    _pad1: f32,
    local_offset: vec3f,
    _pad2: f32,
    local_rotation: vec4f,
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

struct ManifoldPoint {
    position: vec3f,
    depth: f32,
    accumulated_normal: f32,
    accumulated_tangent_1: f32,
    accumulated_tangent_2: f32,
    _pad0: f32,
}

struct Contact {
    a: u32,
    b: u32,
    point_count: u32,
    _pad0: u32,
    normal: vec3f,
    _pad1: f32,
    points: array<ManifoldPoint, CONTACT_MAX_POINTS>,
}

struct Constraint {
    kind: u32,
    a: u32,
    b: u32,
    _pad0: u32,
    anchor_a: vec3f,
    _pad1: f32,
    anchor_b: vec3f,
    _pad2: f32,
    axis_a: vec3f,
    _pad3: f32,
    axis_b: vec3f,
    _pad4: f32,
    distance: f32,
    _pad5: f32,
    _pad6: f32,
    _pad7: f32,
    accumulated: array<f32, 8>,
}

struct Query {
    origin: vec3f,
    kind: u32,
    direction: vec3f,
    extent: f32,
    slot: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct QueryResult {
    body_id: u32,
    body_generation: u32,
    distance: f32,
    hit: u32,
    point: vec3f,
    _pad0: f32,
    normal: vec3f,
    _pad1: f32,
}

fn quat_mul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(
        a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz),
    );
}

fn quat_conjugate(q: vec4f) -> vec4f {
    return vec4f(-q.xyz, q.w);
}

fn quat_rotate(q: vec4f, v: vec3f) -> vec3f {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

fn normalize_quat(q: vec4f) -> vec4f {
    let n = length(q);
    return vec4f(q.xyz / n, q.w / n);
}

fn sign_normalize(v: vec3f) -> vec3f {
    let n = length(v);
    if (n > 1e-8) {
        return v / n;
    }
    return vec3f(0.0, 1.0, 0.0);
}

struct TangentBasis {
    first: vec3f,
    second: vec3f,
}

fn make_tangents(normal: vec3f) -> TangentBasis {
    var tangent = vec3f(1.0, 0.0, 0.0);
    if (abs(normal.x) > 0.9) {
        tangent = vec3f(0.0, 1.0, 0.0);
    }
    let t1 = normalize(cross(normal, tangent));
    let t2 = cross(normal, t1);
    return TangentBasis(t1, t2);
}

fn apply_inverse_inertia(body: RigidBody, v: vec3f) -> vec3f {
    let q = body.orientation;
    let local = quat_rotate(quat_conjugate(q), v);
    return quat_rotate(q, body.inverse_inertia_body * local);
}

fn world_collider(body: RigidBody, collider: Collider) -> Collider {
    var world = collider;
    world.local_offset = quat_rotate(body.orientation, collider.local_offset) + body.position;
    world.local_rotation = quat_mul(body.orientation, collider.local_rotation);
    return world;
}

fn relative_velocity(body_a: RigidBody, body_b: RigidBody, point_a: vec3f, point_b: vec3f) -> vec3f {
    let va = body_a.velocity + cross(body_a.angular_velocity, point_a - body_a.position);
    let vb = body_b.velocity + cross(body_b.angular_velocity, point_b - body_b.position);
    return vb - va;
}

fn point_momentum_mass(body_a: RigidBody, body_b: RigidBody, point_a: vec3f, point_b: vec3f, axis: vec3f) -> f32 {
    let ra = point_a - body_a.position;
    let rb = point_b - body_b.position;
    let rax = cross(ra, axis);
    let rbx = cross(rb, axis);
    return body_a.inverse_mass
        + body_b.inverse_mass
        + dot(rax, apply_inverse_inertia(body_a, rax))
        + dot(rbx, apply_inverse_inertia(body_b, rbx));
}

fn apply_pair_impulse(
    body_a: ptr<function, RigidBody>,
    body_b: ptr<function, RigidBody>,
    point_a: vec3f,
    point_b: vec3f,
    impulse: vec3f,
) {
    (*body_a).velocity = (*body_a).velocity - impulse * (*body_a).inverse_mass;
    (*body_a).angular_velocity = (*body_a).angular_velocity
        - apply_inverse_inertia(*body_a, cross(point_a - (*body_a).position, impulse));
    (*body_b).velocity = (*body_b).velocity + impulse * (*body_b).inverse_mass;
    (*body_b).angular_velocity = (*body_b).angular_velocity
        + apply_inverse_inertia(*body_b, cross(point_b - (*body_b).position, impulse));
}

struct Segment {
    start: vec3f,
    end: vec3f,
}

fn box_center(body: RigidBody, collider: Collider) -> vec3f {
    return body.position + quat_rotate(body.orientation, collider.local_offset);
}

fn box_rotated_axes(body: RigidBody, collider: Collider) -> array<vec3f, 3> {
    let q = quat_mul(body.orientation, collider.local_rotation);
    let axes: array<vec3f, 3> = array(
        quat_rotate(q, vec3f(1.0, 0.0, 0.0)),
        quat_rotate(q, vec3f(0.0, 1.0, 0.0)),
        quat_rotate(q, vec3f(0.0, 0.0, 1.0)),
    );
    return axes;
}

fn closest_point_box(point: vec3f, body: RigidBody, collider: Collider) -> vec3f {
    let q = quat_mul(body.orientation, collider.local_rotation);
    let local = quat_rotate(quat_conjugate(q), point - box_center(body, collider));
    let clamped = clamp(local, -collider.half_extents, collider.half_extents);
    return box_center(body, collider) + quat_rotate(q, clamped);
}

fn closest_point_segment(point: vec3f, a: vec3f, b: vec3f) -> vec3f {
    let ab = b - a;
    let denom = dot(ab, ab);
    if (denom < 1e-10) {
        return a;
    }
    let t = clamp(dot(point - a, ab) / denom, 0.0, 1.0);
    return a + ab * t;
}

fn largest_axis(v: vec3f) -> u32 {
    let av = abs(v);
    if (av.x > av.y && av.x > av.z) {
        return 0u;
    }
    if (av.y > av.z) {
        return 1u;
    }
    return 2u;
}

