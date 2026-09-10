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
    max_cells_per_collider: u32,
    dynamic_count: u32,
    sleep_velocity: f32,
    sleep_angular_velocity: f32,
    sleep_time: f32,
    wake_velocity: f32,
    friction_combine: u32,
    restitution_combine: u32,
    tempering: f32,
    _pad6: f32,
    event_slot: u32,
}

struct BodyState {
    position: vec3f,
    _pad0: f32,
    prev_position: vec3f,
    _pad1: f32,
    orientation: vec4f,
    velocity: vec3f,
    _pad2: f32,
    angular_velocity: vec3f,
    _pad3: f32,
    force: vec3f,
    _pad4: f32,
    torque: vec3f,
    _pad5: f32,
    body_id: u32,
    generation: u32,
    sleep_timer: f32,
    sleeping: u32,
}

struct BodyDescriptor {
    inverse_mass: f32,
    linear_damping: f32,
    angular_damping: f32,
    gravity_scale: f32,
    sleep_velocity: f32,
    sleep_angular_velocity: f32,
    flags: u32,
    _pad0: u32,
    collision_group: u32,
    collision_mask: u32,
    _pad1: u32,
    _pad4: u32,
    com: vec3f,
    _pad2: f32,
    inverse_inertia: array<f32, 6>,
    _pad3: array<f32, 2>,
}

/// A host-owned descriptor paired with a device-owned state row.
struct Body {
    state: BodyState,
    desc: BodyDescriptor,
}

struct Collider {
    kind: u32,
    flags: u32,
    radius: f32,
    half_height: f32,
    half_extents: vec3f,
    collision_group: u32,
    local_offset: vec3f,
    collision_mask: u32,
    local_rotation: vec4f,
    friction: f32,
    restitution: f32,
    source: u32,
    rolling_friction: f32,
    scale: vec3f,
    spin_friction: f32,
}

struct ShapeSource {
    kind: u32,
    vertex_offset: u32,
    vertex_count: u32,
    triangle_offset: u32,
    triangle_count: u32,
    node_offset: u32,
    node_count: u32,
    _pad: u32,
}

struct BvhNode {
    min: vec3f,
    _pad0: f32,
    max: vec3f,
    _pad1: f32,
    left: u32,
    right: u32,
    leaf: u32,
    _pad2: u32,
}

struct Triangle {
    a: u32,
    b: u32,
    c: u32,
    _pad0: u32,
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
    sensor: u32,
    first_body_id: u32,
    second_body_id: u32,
    first_generation: u32,
    second_generation: u32,
    normal: vec3f,
    events: u32,
    friction: f32,
    restitution: f32,
    rolling_friction: f32,
    spin_friction: f32,
    points: array<ManifoldPoint, CONTACT_MAX_POINTS>,
}

struct ConstraintDescriptor {
    kind: u32,
    a: u32,
    b: u32,
    flags: u32,
    anchor_a: vec3f,
    _pad1: f32,
    anchor_b: vec3f,
    _pad2: f32,
    axis_a: vec3f,
    _pad3: f32,
    axis_b: vec3f,
    _pad4: f32,
    distance: f32,
    limit_min: f32,
    limit_max: f32,
    swing_a: f32,
    swing_b: f32,
    motor_speed: f32,
    motor_max_force: f32,
    spring_frequency: f32,
    spring_damping_ratio: f32,
    break_force: f32,
    break_torque: f32,
    gear_ratio: f32,
    pulley_fixed_a: vec3f,
    _pad_pulley_a: f32,
    pulley_fixed_b: vec3f,
    _pad_pulley_b: f32,
    motor_target: f32,
    motor_stiffness: f32,
    motor_damping: f32,
    cone_angle: f32,
    reference: vec4f,
    linear_limit_min: vec3f,
    _pad_lim_min: f32,
    linear_limit_max: vec3f,
    _pad_lim_max: f32,
    angular_limit_min: vec3f,
    _pad_ang_min: f32,
    angular_limit_max: vec3f,
    _pad_ang_max: f32,
    linear_motor_target: vec3f,
    _pad_lin_target: f32,
    linear_motor_stiffness: vec3f,
    _pad_lin_stiff: f32,
    linear_motor_damping: vec3f,
    _pad_lin_damp: f32,
    angular_motor_target: vec3f,
    _pad_ang_target: f32,
    angular_motor_stiffness: vec3f,
    _pad_ang_stiff: f32,
    angular_motor_damping: vec3f,
    _pad_ang_damp: f32,
    linear_motor_force: vec3f,
    _pad_lin_force: f32,
    angular_motor_force: vec3f,
    _pad_ang_force: f32,
}

/// Accumulated impulses of a constraint slot, owned by the device.
struct ConstraintRuntime {
    accumulated: array<f32, 8>,
    broken: u32,
    constraint_id: u32,
    generation: u32,
    _pad0: u32,
}

struct Query {
    kind: u32,
    shape_kind: u32,
    filter_flags: u32,
    slot: u32,
    group: u32,
    mask: u32,
    source: u32,
    max_hits: u32,
    exclude_id: u32,
    exclude_generation: u32,
    include_id: u32,
    include_generation: u32,
    origin: vec3f,
    _pad0: f32,
    direction: vec3f,
    extent: f32,
    radius: f32,
    half_height: f32,
    _pad1: f32,
    _pad2: f32,
    half_extents: vec3f,
    _pad3: f32,
    orientation: vec4f,
}

struct QueryResultHeader {
    count: atomic<u32>,
    overflow: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
}

struct QueryHit {
    body_id: u32,
    body_generation: u32,
    distance: f32,
    collider_index: u32,
    point: vec3f,
    _pad1: f32,
    normal: vec3f,
    _pad2: f32,
}

struct QueryResult {
    header: QueryResultHeader,
    hits: array<QueryHit, MAX_HITS_PER_QUERY>,
}

struct ContactEvent {
    kind: u32,
    sensor: u32,
    first_id: u32,
    first_generation: u32,
    second_id: u32,
    second_generation: u32,
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

fn body_is_inert(body: Body) -> bool {
    return body.desc.inverse_mass == 0.0 || body.state.sleeping != 0u;
}

fn body_frozen(body: Body) -> Body {
    var frozen = body;
    frozen.desc.inverse_mass = 0.0;
    frozen.desc.inverse_inertia = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    return frozen;
}

fn body_is_dynamic(body: Body) -> bool {
    return body.desc.inverse_mass > 0.0 && (body.desc.flags & BODY_KINEMATIC) == 0u;
}

fn body_is_kinematic(body: Body) -> bool {
    return (body.desc.flags & BODY_KINEMATIC) != 0u;
}

fn body_has_ccd(body: Body) -> bool {
    return (body.desc.flags & BODY_CCD) != 0u;
}

fn body_is_static(body: Body) -> bool {
    return body.desc.inverse_mass == 0.0 && (body.desc.flags & BODY_KINEMATIC) == 0u;
}

fn collider_is_sensor(collider: Collider) -> bool {
    return (collider.flags & COLLIDER_SENSOR) != 0u;
}

fn collider_filter_intersects(
    first_body: Body, first_collider: Collider,
    second_body: Body, second_collider: Collider,
) -> bool {
    let first_group = select(first_body.desc.collision_group, first_collider.collision_group, first_collider.collision_group != NO_COLLISION_FILTER);
    let first_mask = select(first_body.desc.collision_mask, first_collider.collision_mask, first_collider.collision_mask != NO_COLLISION_FILTER);
    let second_group = select(second_body.desc.collision_group, second_collider.collision_group, second_collider.collision_group != NO_COLLISION_FILTER);
    let second_mask = select(second_body.desc.collision_mask, second_collider.collision_mask, second_collider.collision_mask != NO_COLLISION_FILTER);
    return (first_group & second_mask) != 0u && (second_group & first_mask) != 0u;
}

fn collider_filter_query(query: Query, body: Body, collider: Collider) -> bool {
    let group = select(body.desc.collision_group, collider.collision_group, collider.collision_group != NO_COLLISION_FILTER);
    let mask = select(body.desc.collision_mask, collider.collision_mask, collider.collision_mask != NO_COLLISION_FILTER);
    return query.group == 0u || ((group & query.mask) != 0u && (query.group & mask) != 0u);
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

fn material_combine(first: f32, second: f32, mode: u32) -> f32 {
    if (mode == 1u) {
        return min(first, second);
    }
    if (mode == 2u) {
        return max(first, second);
    }
    if (mode == 3u) {
        return (first + second) * 0.5;
    }
    return sqrt(first * second);
}

fn body_com(body: Body) -> vec3f {
    return body_com_of(body.state, body.desc);
}

fn body_com_of(state: BodyState, desc: BodyDescriptor) -> vec3f {
    return state.position + quat_rotate(state.orientation, desc.com);
}

fn inverse_inertia_local(desc: BodyDescriptor, v: vec3f) -> vec3f {
    let i = desc.inverse_inertia;
    return vec3f(
        i[0] * v.x + i[1] * v.y + i[2] * v.z,
        i[1] * v.x + i[3] * v.y + i[4] * v.z,
        i[2] * v.x + i[4] * v.y + i[5] * v.z,
    );
}

fn apply_inverse_inertia_of(desc: BodyDescriptor, q: vec4f, v: vec3f) -> vec3f {
    let local = quat_rotate(quat_conjugate(q), v);
    return quat_rotate(q, inverse_inertia_local(desc, local));
}

fn apply_inverse_inertia(body: Body, v: vec3f) -> vec3f {
    return apply_inverse_inertia_of(body.desc, body.state.orientation, v);
}

fn relative_velocity(body_a: Body, body_b: Body, point_a: vec3f, point_b: vec3f) -> vec3f {
    let va = body_a.state.velocity + cross(body_a.state.angular_velocity, point_a - body_com(body_a));
    let vb = body_b.state.velocity + cross(body_b.state.angular_velocity, point_b - body_com(body_b));
    return vb - va;
}

fn point_momentum_mass(body_a: Body, body_b: Body, point_a: vec3f, point_b: vec3f, axis: vec3f) -> f32 {
    let ra = point_a - body_com(body_a);
    let rb = point_b - body_com(body_b);
    let rax = cross(ra, axis);
    let rbx = cross(rb, axis);
    return body_a.desc.inverse_mass
        + body_b.desc.inverse_mass
        + dot(rax, apply_inverse_inertia(body_a, rax))
        + dot(rbx, apply_inverse_inertia(body_b, rbx));
}

fn apply_pair_impulse(
    body_a: ptr<function, Body>,
    body_b: ptr<function, Body>,
    point_a: vec3f,
    point_b: vec3f,
    impulse: vec3f,
) {
    (*body_a).state.velocity = (*body_a).state.velocity - impulse * (*body_a).desc.inverse_mass;
    (*body_a).state.angular_velocity = (*body_a).state.angular_velocity
        - apply_inverse_inertia(*body_a, cross(point_a - body_com(*body_a), impulse));
    (*body_b).state.velocity = (*body_b).state.velocity + impulse * (*body_b).desc.inverse_mass;
    (*body_b).state.angular_velocity = (*body_b).state.angular_velocity
        + apply_inverse_inertia(*body_b, cross(point_b - body_com(*body_b), impulse));
}

struct Segment {
    start: vec3f,
    end: vec3f,
}

struct WorldShape {
    kind: u32,
    radius: f32,
    half_height: f32,
    center: vec3f,
    half_extents: vec3f,
    rotation: vec4f,
    source: u32,
    scale: vec3f,
}

fn world_collider(state: BodyState, collider: Collider) -> WorldShape {
    var world: WorldShape;
    world.kind = collider.kind;
    world.radius = collider.radius;
    world.half_height = collider.half_height;
    world.center = state.position + quat_rotate(state.orientation, collider.local_offset);
    world.half_extents = collider.half_extents;
    world.rotation = quat_mul(state.orientation, collider.local_rotation);
    world.source = collider.source;
    world.scale = collider.scale;
    return world;
}

fn shape_axis(world: WorldShape) -> vec3f {
    return quat_rotate(world.rotation, vec3f(0.0, 1.0, 0.0));
}

fn support_local(world: WorldShape, direction: vec3f) -> vec3f {
    let d = direction;
    let n = normalize(direction);
    if (world.kind == SHAPE_SPHERE) {
        return n * world.radius;
    }
    if (world.kind == SHAPE_CUBOID) {
        let signs = select(vec3f(-1.0), vec3f(1.0), d > vec3f(0.0));
        return signs * world.half_extents;
    }
    if (world.kind == SHAPE_CAPSULE) {
        let tip = vec3f(0.0, 1.0, 0.0) * world.half_height * select(-1.0, 1.0, d.y > 0.0);
        return tip + n * world.radius;
    }
    if (world.kind == SHAPE_CYLINDER) {
        let base = vec3f(0.0, 1.0, 0.0) * world.half_height * select(-1.0, 1.0, d.y > 0.0);
        let radial = d - vec3f(0.0, d.y, 0.0);
        if (length(radial) < 1e-6) {
            return base;
        }
        return base + normalize(radial) * world.radius;
    }
    if (world.kind == SHAPE_TRIANGLE) {
        let source = shape_sources[world.source];
        let tri = shape_triangles[source.triangle_offset + u32(world.radius)];
        let p0 = shape_vertices[source.vertex_offset + tri.a].xyz;
        let p1 = shape_vertices[source.vertex_offset + tri.b].xyz;
        let p2 = shape_vertices[source.vertex_offset + tri.c].xyz;
        let s0 = dot(p0, d);
        let s1 = dot(p1, d);
        let s2 = dot(p2, d);
        if (s0 >= s1 && s0 >= s2) {
            return p0;
        }
        if (s1 >= s2) {
            return p1;
        }
        return p2;
    }
    let source = shape_sources[world.source];
    var best_dot = -3.402823466e38;
    var best = vec3f(0.0);
    for (var i = 0u; i < source.vertex_count; i = i + 1u) {
        let v = shape_vertices[source.vertex_offset + i].xyz;
        let s = dot(v, d);
        if (s > best_dot) {
            best_dot = s;
            best = v;
        }
    }
    return best;
}

fn support(world: WorldShape, direction: vec3f) -> vec3f {
    let local_d = quat_rotate(quat_conjugate(world.rotation), direction);
    let scaled_d = local_d * world.scale;
    let local = support_local(world, scaled_d);
    return world.center + quat_rotate(world.rotation, local * world.scale);
}

fn triangle_vertex(world: WorldShape, index: u32) -> vec3f {
    let source = shape_sources[world.source];
    let local = shape_vertices[source.vertex_offset + index].xyz;
    return world.center + quat_rotate(world.rotation, local * world.scale);
}

fn support_triangle(world: WorldShape, direction: vec3f) -> vec3f {
    let a = triangle_vertex(world, 0u);
    let b = triangle_vertex(world, 1u);
    let c = triangle_vertex(world, 2u);
    var best = a;
    var best_dot = dot(a, direction);
    let b_dot = dot(b, direction);
    if (b_dot > best_dot) {
        best = b;
        best_dot = b_dot;
    }
    let c_dot = dot(c, direction);
    if (c_dot > best_dot) {
        best = c;
    }
    return best;
}

struct SimplexPoint {
    w: vec3f,
    a: vec3f,
    b: vec3f,
}

struct SimplexResult {
    v: vec3f,
    lambdas: vec4f,
    penetrating: bool,
    point_a: vec3f,
    point_b: vec3f,
}

fn closest_on_triangle(origin: vec3f, a: vec3f, b: vec3f, c: vec3f, out_weights: ptr<function, vec3f>) -> vec3f {
    let ab = b - a;
    let ac = c - a;
    let ap = origin - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        *out_weights = vec3f(1.0, 0.0, 0.0);
        return a;
    }
    let bp = origin - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        *out_weights = vec3f(0.0, 1.0, 0.0);
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let t = d1 / (d1 - d3);
        *out_weights = vec3f(1.0 - t, t, 0.0);
        return a + ab * t;
    }
    let cp = origin - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        *out_weights = vec3f(0.0, 0.0, 1.0);
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let t = d2 / (d2 - d6);
        *out_weights = vec3f(1.0 - t, 0.0, t);
        return a + ac * t;
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        *out_weights = vec3f(0.0, 1.0 - t, t);
        return b + (c - b) * t;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    *out_weights = vec3f(1.0 - v - w, v, w);
    return a + ab * v + ac * w;
}

fn tetra_contains(tet: array<SimplexPoint, 4>, origin: vec3f) -> bool {
    for (var face = 0u; face < 4u; face = face + 1u) {
        var a: vec3f;
        var b: vec3f;
        var c: vec3f;
        if (face == 0u) {
            a = tet[0].w;
            b = tet[1].w;
            c = tet[2].w;
        } else if (face == 1u) {
            a = tet[0].w;
            b = tet[2].w;
            c = tet[3].w;
        } else if (face == 2u) {
            a = tet[0].w;
            b = tet[3].w;
            c = tet[1].w;
        } else {
            a = tet[1].w;
            b = tet[3].w;
            c = tet[2].w;
        }
        let normal = cross(b - a, c - a);
        if (dot(normal, origin - a) > 0.0) {
            return false;
        }
    }
    return true;
}

fn simplex_closest(simplex: array<SimplexPoint, 4>, count: u32) -> SimplexResult {
    var result: SimplexResult;
    result.v = vec3f(3.402823466e38);
    result.lambdas = vec4f(0.0);
    result.penetrating = count == 4u && tetra_contains(simplex, vec3f(0.0));
    result.point_a = vec3f(0.0);
    result.point_b = vec3f(0.0);
    if (result.penetrating) {
        result.v = vec3f(0.0);
        return result;
    }
    for (var i = 0u; i < count; i = i + 1u) {
        let d = length(simplex[i].w);
        if (d < length(result.v)) {
            result.v = simplex[i].w;
            result.lambdas = vec4f(0.0);
            result.lambdas[i] = 1.0;
        }
    }
    for (var i = 0u; i < count; i = i + 1u) {
        for (var j = i + 1u; j < count; j = j + 1u) {
            let a = simplex[i].w;
            let b = simplex[j].w;
            let ab = b - a;
            let denom = dot(ab, ab);
            if (denom < 1e-12) {
                continue;
            }
            let t = clamp(-dot(a, ab) / denom, 0.0, 1.0);
            let point = a + ab * t;
            if (length(point) < length(result.v)) {
                result.v = point;
                result.lambdas = vec4f(0.0);
                result.lambdas[i] = 1.0 - t;
                result.lambdas[j] = t;
            }
        }
    }
    if (count >= 3u) {
        for (var i = 0u; i < count; i = i + 1u) {
            for (var j = i + 1u; j < count; j = j + 1u) {
                for (var k = j + 1u; k < count; k = k + 1u) {
                    var weights = vec3f(0.0);
                    let point = closest_on_triangle(vec3f(0.0), simplex[i].w, simplex[j].w, simplex[k].w, &weights);
                    if (length(point) < length(result.v)) {
                        result.v = point;
                        result.lambdas = vec4f(0.0);
                        result.lambdas[i] = weights.x;
                        result.lambdas[j] = weights.y;
                        result.lambdas[k] = weights.z;
                    }
                }
            }
        }
    }
    for (var i = 0u; i < count; i = i + 1u) {
        result.point_a = result.point_a + result.lambdas[i] * simplex[i].a;
        result.point_b = result.point_b + result.lambdas[i] * simplex[i].b;
    }
    return result;
}


fn support_pair(first: WorldShape, second: WorldShape, direction: vec3f) -> SimplexPoint {
    let a = support(first, direction);
    let b = support(second, -direction);
    return SimplexPoint(a - b, a, b);
}

struct EpaPoint {
    w: vec3f,
    a: vec3f,
    b: vec3f,
}

struct EpaFace {
    a: u32,
    b: u32,
    c: u32,
    normal: vec3f,
}

struct EpaResult {
    valid: bool,
    normal: vec3f,
    depth: f32,
    point: vec3f,
}

fn epa_tetrahedron(first: WorldShape, second: WorldShape, simplex: array<SimplexPoint, 4>, count: u32) -> EpaResult {
    var polytope: array<EpaPoint, 64>;
    var new_faces: array<EpaFace, 64>;
    var faces: array<EpaFace, 128>;
    var visible: array<u32, 128>;
    var vertex_count = 0u;
    for (var i = 0u; i < count; i = i + 1u) {
        polytope[vertex_count] = EpaPoint(simplex[i].w, simplex[i].a, simplex[i].b);
        vertex_count = vertex_count + 1u;
    }
    var face_count = 0u;
    if (vertex_count != 4u) {
        var invalid: EpaResult;
        invalid.valid = false;
        invalid.normal = vec3f(0.0, 1.0, 0.0);
        invalid.depth = 0.0;
        invalid.point = vec3f(0.0);
        return invalid;
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        var a: u32;
        var b: u32;
        var c: u32;
        if (i == 0u) {
            a = 0u;
            b = 1u;
            c = 2u;
        } else if (i == 1u) {
            a = 0u;
            b = 2u;
            c = 3u;
        } else if (i == 2u) {
            a = 0u;
            b = 3u;
            c = 1u;
        } else {
            a = 1u;
            b = 3u;
            c = 2u;
        }
        let normal = cross(polytope[b].w - polytope[a].w, polytope[c].w - polytope[a].w);
        let length_n = length(normal);
        if (length_n < 1e-10) {
            var invalid: EpaResult;
            invalid.valid = false;
            invalid.normal = vec3f(0.0, 1.0, 0.0);
            invalid.depth = 0.0;
            invalid.point = vec3f(0.0);
            return invalid;
        }
        let unit = normal / length_n;
        let centroid = (polytope[a].w + polytope[b].w + polytope[c].w) * (1.0 / 3.0);
        faces[face_count] = EpaFace(a, b, c, select(unit, -unit, dot(unit, centroid) > 0.0));
        face_count = face_count + 1u;
    }
    var best_dist = 0.0;
    var best_face = 0u;
    for (var iter = 0u; iter < 32u; iter = iter + 1u) {
        best_dist = 3.402823466e38;
        best_face = 0u;
        for (var i = 0u; i < face_count; i = i + 1u) {
            let dist = dot(faces[i].normal, polytope[faces[i].a].w);
            if (dist < best_dist && length(faces[i].normal) > 0.5) {
                best_dist = dist;
                best_face = i;
            }
        }
        let probe = support_pair(first, second, faces[best_face].normal).w;
        let gain = dot(probe, faces[best_face].normal) - best_dist;
        if (gain < 1e-4) {
            break;
        }
        if (vertex_count >= 64u) {
            break;
        }
        polytope[vertex_count] = EpaPoint(probe, support(first, faces[best_face].normal), support(second, -faces[best_face].normal));
        vertex_count = vertex_count + 1u;
        var visible_count = 0u;
        for (var i = 0u; i < face_count; i = i + 1u) {
            let vert = polytope[faces[i].a].w;
            if (dot(faces[i].normal, probe - vert) > 1e-6) {
                visible[visible_count] = i;
                visible_count = visible_count + 1u;
            }
        }
        var new_count = 0u;
        for (var i = 0u; i < visible_count; i = i + 1u) {
            let face = faces[visible[i]];
            var edges: array<vec2f, 3>;
            edges[0] = vec2f(f32(face.a), f32(face.b));
            edges[1] = vec2f(f32(face.b), f32(face.c));
            edges[2] = vec2f(f32(face.c), f32(face.a));
            for (var e = 0u; e < 3u; e = e + 1u) {
                let start = u32(edges[e].x);
                let end = u32(edges[e].y);
                var is_shared = false;
                for (var j = 0u; j < visible_count; j = j + 1u) {
                    if (j == i) {
                        continue;
                    }
                    let other = faces[visible[j]];
                    if ((other.a == end && other.b == start) || (other.b == end && other.c == start) || (other.c == end && other.a == start)) {
                        is_shared = true;
                    }
                }
                if (!is_shared && new_count < 64u) {
                    let apex = u32(vertex_count - 1u);
                    let normal = cross(polytope[end].w - polytope[apex].w, polytope[start].w - polytope[apex].w);
                    let length_n = length(normal);
                    if (length_n > 1e-10) {
                        let unit = normal / length_n;
                        let centroid = (polytope[apex].w + polytope[start].w + polytope[end].w) * (1.0 / 3.0);
                        new_faces[new_count] = EpaFace(apex, start, end, select(unit, -unit, dot(unit, centroid) > 0.0));
                        new_count = new_count + 1u;
                    }
                }
            }
        }
        var write = 0u;
        var seen = 0u;
        for (var i = 0u; i < face_count; i = i + 1u) {
            if (seen < visible_count && visible[seen] == i) {
                seen = seen + 1u;
                continue;
            }
            faces[write] = faces[i];
            write = write + 1u;
        }
        face_count = write;
        if (face_count + new_count > 128u) {
            break;
        }
        for (var i = 0u; i < new_count; i = i + 1u) {
            faces[face_count] = new_faces[i];
            face_count = face_count + 1u;
        }
    }
    best_dist = 3.402823466e38;
    for (var i = 0u; i < face_count; i = i + 1u) {
        let dist = dot(faces[i].normal, polytope[faces[i].a].w);
        if (dist < best_dist && length(faces[i].normal) > 0.5) {
            best_dist = dist;
            best_face = i;
        }
    }
    var result: EpaResult;
    result.valid = best_dist < 3.402823466e38;
    result.normal = faces[best_face].normal;
    result.depth = best_dist;
    result.point = polytope[faces[best_face].a].a;
    return result;
}

fn shape_scale(world: WorldShape) -> f32 {
    var scale = world.radius + world.half_height;
    scale = max(scale, max(max(world.half_extents.x, world.half_extents.y), world.half_extents.z));
    return max(scale, 1e-4) * max(max(world.scale.x, world.scale.y), world.scale.z);
}

struct ConvexClosest {
    distance: f32,
    point_a: vec3f,
    point_b: vec3f,
    normal: vec3f,
    penetrating: bool,
}

fn simplex_expand_dir(simplex: array<SimplexPoint, 4>, count: u32) -> vec3f {
    if (count == 1u) {
        return vec3f(1.0, 0.0, 0.0);
    }
    if (count == 2u) {
        let edge = simplex[1].w - simplex[0].w;
        var d = cross(edge, vec3f(1.0, 0.0, 0.0));
        if (length(d) < 1e-6) {
            d = cross(edge, vec3f(0.0, 1.0, 0.0));
        }
        return normalize(d);
    }
    let normal = cross(simplex[1].w - simplex[0].w, simplex[2].w - simplex[0].w);
    if (length(normal) > 1e-8) {
        return normalize(normal);
    }
    return vec3f(0.0, 1.0, 0.0);
}

fn simplex_reduce(simplex: array<SimplexPoint, 4>, count: u32, result: SimplexResult, best_type: u32, best_i: u32, best_j: u32, best_k: u32, out_simplex: ptr<function, array<SimplexPoint, 4>>) -> u32 {
    if (best_type == 1u) {
        *out_simplex = simplex;
        (*out_simplex)[0u] = simplex[best_i];
        return 1u;
    }
    if (best_type == 2u) {
        *out_simplex = simplex;
        (*out_simplex)[0u] = simplex[best_i];
        (*out_simplex)[1u] = simplex[best_j];
        return 2u;
    }
    if (best_type == 3u) {
        *out_simplex = simplex;
        (*out_simplex)[0u] = simplex[best_i];
        (*out_simplex)[1u] = simplex[best_j];
        (*out_simplex)[2u] = simplex[best_k];
        return 3u;
    }
    return count;
}

fn convex_closest(first: WorldShape, second: WorldShape, out_simplex: ptr<function, array<SimplexPoint, 4>>, out_count: ptr<function, u32>) -> ConvexClosest {
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    var result: ConvexClosest;
    result.distance = 0.0;
    result.point_a = first.center;
    result.point_b = second.center;
    result.normal = sign_normalize(second.center - first.center);
    result.penetrating = false;
    var dir = second.center - first.center;
    if (length(dir) < 1e-8) {
        dir = vec3f(1.0, 0.0, 0.0);
    }
    simplex[0] = support_pair(first, second, dir);
    count = 1u;
    var closest = simplex_closest(simplex, count);
    let tiny = 1e-4 * min(shape_scale(first), shape_scale(second));
    for (var iter = 0u; iter < 24u; iter = iter + 1u) {
        if (closest.penetrating || length(closest.v) < tiny) {
            while (count < 4u) {
                let dir = simplex_expand_dir(simplex, count);
                simplex[count] = support_pair(first, second, dir);
                count = count + 1u;
            }
            result.penetrating = true;
            result.distance = 0.0;
            result.point_a = closest.point_a;
            result.point_b = closest.point_b;
            break;
        }
        let sp = support_pair(first, second, -closest.v);
        if (dot(sp.w, -closest.v) <= dot(closest.v, -closest.v) + 1e-4 * max(1.0, dot(closest.v, closest.v))) {
            result.distance = length(closest.v);
            result.point_a = closest.point_a;
            result.point_b = closest.point_b;
            if (result.distance > 1e-6) {
                result.normal = sign_normalize(closest.point_b - closest.point_a);
            }
            break;
        }
        if (count == 4u) {
            result.distance = length(closest.v);
            result.point_a = closest.point_a;
            result.point_b = closest.point_b;
            if (result.distance > 1e-6) {
                result.normal = sign_normalize(closest.point_b - closest.point_a);
            }
            break;
        }
        simplex[count] = sp;
        count = count + 1u;
        closest = simplex_closest(simplex, count);
    }
    *out_simplex = simplex;
    *out_count = count;
    return result;
}

struct ShapeHit {
    distance: f32,
    point: vec3f,
    normal: vec3f,
}

fn no_hit() -> ShapeHit {
    return ShapeHit(NO_HIT, vec3f(0.0), vec3f(0.0));
}

fn support_projection_depth(first: WorldShape, second: WorldShape, direction: vec3f) -> f32 {
    let n = normalize(direction);
    return dot(support(first, n) - support(second, -n), n);
}

fn convex_penetration_probe(first: WorldShape, second: WorldShape, simplex: array<SimplexPoint, 4>, count: u32) -> EpaResult {
    var best: EpaResult;
    best.valid = false;
    best.normal = vec3f(0.0, 1.0, 0.0);
    best.depth = 3.402823466e38;
    best.point = (first.center + second.center) * 0.5;
    var probed: array<vec3f, 16>;
    var probe_count = 0u;
    for (var i = 0u; i < count; i = i + 1u) {
        for (var j = i + 1u; j < count; j = j + 1u) {
            for (var k = j + 1u; k < count; k = k + 1u) {
                let normal = cross(simplex[j].w - simplex[i].w, simplex[k].w - simplex[i].w);
                if (length(normal) > 1e-8 && probe_count < 16u) {
                    probed[probe_count] = normal;
                    probe_count = probe_count + 1u;
                }
            }
        }
    }
    let center_dir = second.center - first.center;
    if (length(center_dir) > 1e-8 && probe_count < 16u) {
        probed[probe_count] = center_dir;
        probe_count = probe_count + 1u;
    }
    let axes: array<vec3f, 6> = array(
        vec3f(1.0, 0.0, 0.0),
        vec3f(-1.0, 0.0, 0.0),
        vec3f(0.0, 1.0, 0.0),
        vec3f(0.0, -1.0, 0.0),
        vec3f(0.0, 0.0, 1.0),
        vec3f(0.0, 0.0, -1.0),
    );
    for (var i = 0u; i < 6u; i = i + 1u) {
        if (probe_count < 16u) {
            probed[probe_count] = axes[i];
            probe_count = probe_count + 1u;
        }
    }
    for (var i = 0u; i < probe_count; i = i + 1u) {
        let depth = support_projection_depth(first, second, probed[i]);
        if (depth < best.depth) {
            best.depth = depth;
            best.normal = normalize(probed[i]);
        }
    }
    best.valid = best.depth < 3.402823466e38;
    let point_a = support(first, best.normal);
    let point_b = support(second, -best.normal);
    best.point = (point_a + point_b) * 0.5;
    return best;
}

fn convex_hit(first: WorldShape, second: WorldShape) -> ShapeHit {
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    let closest = convex_closest(first, second, &simplex, &count);
    if (!closest.penetrating) {
        let probe = convex_penetration_probe(first, second, simplex, count);
        if (probe.depth > 0.0) {
            var normal = probe.normal;
            if (dot(normal, second.center - first.center) < 0.0) {
                normal = -normal;
            }
            return ShapeHit(-probe.depth, (closest.point_a + closest.point_b) * 0.5, normal);
        }
        if (closest.distance > 0.0) {
            return ShapeHit(closest.distance, (closest.point_a + closest.point_b) * 0.5, closest.normal);
        }
        return no_hit();
    }
    let epa = epa_tetrahedron(first, second, simplex, count);
    var result: ShapeHit;
    if (epa.valid && epa.depth > 0.0 && epa.depth < 3.402823466e38 && length(epa.normal) > 0.5 && epa.depth < support_projection_depth(first, second, epa.normal) + 0.1 * min(shape_scale(first), shape_scale(second))) {
        let point_a = support(first, epa.normal);
        let point_b = support(second, -epa.normal);
        result = ShapeHit(-epa.depth, (point_a + point_b) * 0.5, epa.normal);
    } else {
        let probe = convex_penetration_probe(first, second, simplex, count);
        let point_a = support(first, probe.normal);
        let point_b = support(second, -probe.normal);
        result = ShapeHit(-probe.depth, (point_a + point_b) * 0.5, probe.normal);
    }
    if (dot(result.normal, second.center - first.center) < 0.0) {
        result.normal = -result.normal;
    }
    return result;
}

fn min_radius(collider: Collider) -> f32 {
    if (collider.kind == SHAPE_SPHERE) {
        return collider.radius;
    }
    if (collider.kind == SHAPE_CUBOID) {
        return min(min(collider.half_extents.x, collider.half_extents.y), collider.half_extents.z);
    }
    if (collider.kind == SHAPE_CAPSULE) {
        return collider.radius;
    }
    if (collider.kind == SHAPE_CYLINDER) {
        return min(collider.radius, collider.half_height);
    }
    return 0.0;
}

fn world_aabb_of(world: WorldShape) -> Aabb {
    if (world.kind == SHAPE_PLANE) {
        var plane_box: Aabb;
        plane_box.min = world.center - vec3f(1e6);
        plane_box.max = world.center + vec3f(1e6);
        return plane_box;
    }
    var extent = vec3f(0.0);
    if (world.kind == SHAPE_SPHERE) {
        extent = vec3f(world.radius);
    } else if (world.kind == SHAPE_CUBOID) {
        let e = world.half_extents;
        extent = abs(quat_rotate(world.rotation, vec3f(e.x, 0.0, 0.0)))
            + abs(quat_rotate(world.rotation, vec3f(0.0, e.y, 0.0)))
            + abs(quat_rotate(world.rotation, vec3f(0.0, 0.0, e.z)));
    } else if (world.kind == SHAPE_CAPSULE) {
        let axis = shape_axis(world);
        extent = abs(axis * world.half_height) + vec3f(world.radius);
    } else if (world.kind == SHAPE_CYLINDER) {
        let axis = shape_axis(world);
        extent = abs(axis * world.half_height) + vec3f(world.radius);
    } else {
        let source = shape_sources[world.source];
        var min_v = vec3f(3.402823466e38);
        var max_v = vec3f(-3.402823466e38);
        for (var i = 0u; i < source.vertex_count; i = i + 1u) {
            let local = shape_vertices[source.vertex_offset + i].xyz;
            let p = world.center + quat_rotate(world.rotation, local);
            min_v = min(min_v, p);
            max_v = max(max_v, p);
        }
        extent = (max_v - min_v) * 0.5;
    }
    let s = max(max(world.scale.x, world.scale.y), world.scale.z);
    extent = extent * s;
    var aabb: Aabb;
    aabb.min = world.center - extent;
    aabb.max = world.center + extent;
    return aabb;
}

fn aabb_overlap(first: Aabb, second: Aabb) -> bool {
    return all(first.min <= second.max) && all(second.min <= first.max);
}

fn aabb_ray_hit(box_min: vec3f, box_max: vec3f, origin: vec3f, direction: vec3f, extent: f32) -> f32 {
    let inv = 1.0 / direction;
    var tmin = 0.0;
    var tmax = extent;
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        var t1 = (box_min[axis] - origin[axis]) * inv[axis];
        var t2 = (box_max[axis] - origin[axis]) * inv[axis];
        if (t1 > t2) {
            let tmp = t1;
            t1 = t2;
            t2 = tmp;
        }
        tmin = max(tmin, t1);
        tmax = min(tmax, t2);
        if (tmin > tmax) {
            return NO_HIT;
        }
    }
    return tmin;
}

fn ray_sphere(origin: vec3f, direction: vec3f, extent: f32, center: vec3f, radius: f32) -> ShapeHit {
    let offset = origin - center;
    let projection = dot(offset, direction);
    let discriminant = projection * projection - (dot(offset, offset) - radius * radius);
    if (discriminant < 0.0) {
        return no_hit();
    }
    let root = sqrt(discriminant);
    var t = -projection - root;
    if (t < 0.0) {
        t = -projection + root;
    }
    if (t < 0.0 || t > extent) {
        return no_hit();
    }
    let point = origin + direction * t;
    let normal = normalize(point - center);
    return ShapeHit(t, point, normal);
}

fn ray_box(origin: vec3f, direction: vec3f, extent: f32, center: vec3f, q: vec4f, half_extents: vec3f) -> ShapeHit {
    let local_origin = quat_rotate(quat_conjugate(q), origin - center);
    let local_direction = quat_rotate(quat_conjugate(q), direction);
    let inv = 1.0 / local_direction;
    var tmin = 0.0;
    var tmax = extent;
    var normal_axis = vec3f(0.0);
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        var o = local_origin[axis];
        var d = local_direction[axis];
        var n = vec3f(0.0);
        if (axis == 0u) {
            n = vec3f(1.0, 0.0, 0.0);
        } else if (axis == 1u) {
            n = vec3f(0.0, 1.0, 0.0);
        } else {
            n = vec3f(0.0, 0.0, 1.0);
        }
        let half = half_extents[axis];
        var t1 = (-half - o) * inv[axis];
        var t2 = (half - o) * inv[axis];
        var normal_candidate = n;
        if (t1 > t2) {
            let tmp = t1;
            t1 = t2;
            t2 = tmp;
            normal_candidate = -n;
        }
        if (t1 > tmin) {
            tmin = t1;
            normal_axis = normal_candidate;
        }
        if (t2 < tmax) {
            tmax = t2;
        }
        if (tmin > tmax) {
            return no_hit();
        }
    }
    if (tmin < 0.0 || tmin > extent) {
        return no_hit();
    }
    let local_point = local_origin + local_direction * tmin;
    let point = center + quat_rotate(q, local_point);
    let normal = quat_rotate(q, normal_axis);
    return ShapeHit(tmin, point, normal);
}

fn ray_segment(origin: vec3f, direction: vec3f, extent: f32, seg: Segment, radius: f32) -> ShapeHit {
    var best = no_hit();
    for (var sample = 0u; sample <= 8u; sample = sample + 1u) {
        let t = f32(sample) * (1.0 / 8.0);
        let sphere_center = seg.start + (seg.end - seg.start) * t;
        let hit = ray_sphere(origin, direction, extent, sphere_center, radius);
        if (hit.distance < best.distance) {
            best = hit;
        }
    }
    return best;
}

fn ray_capsule(origin: vec3f, direction: vec3f, extent: f32, center: vec3f, axis: vec3f, half_height: f32, radius: f32) -> ShapeHit {
    let seg = Segment(center - axis * half_height, center + axis * half_height);
    return ray_segment(origin, direction, extent, seg, radius);
}

fn ray_cylinder(origin: vec3f, direction: vec3f, extent: f32, center: vec3f, axis: vec3f, half_height: f32, radius: f32) -> ShapeHit {
    var best = no_hit();
    let seg = Segment(center - axis * half_height, center + axis * half_height);
    for (var sample = 0u; sample <= 4u; sample = sample + 1u) {
        let t = f32(sample) * (1.0 / 4.0);
        let circle_center = seg.start + (seg.end - seg.start) * t;
        let hit = ray_sphere(origin, direction, extent, circle_center, radius);
        if (hit.distance < best.distance) {
            best = hit;
        }
    }
    return best;
}

fn ray_triangle(origin: vec3f, direction: vec3f, extent: f32, a: vec3f, b: vec3f, c: vec3f) -> ShapeHit {
    let edge1 = b - a;
    let edge2 = c - a;
    let p = cross(direction, edge2);
    let det = dot(edge1, p);
    if (abs(det) < 1e-8) {
        return no_hit();
    }
    let inv_det = 1.0 / det;
    let t_vec = origin - a;
    let u = dot(t_vec, p) * inv_det;
    if (u < 0.0 || u > 1.0) {
        return no_hit();
    }
    let q = cross(t_vec, edge1);
    let v = dot(direction, q) * inv_det;
    if (v < 0.0 || u + v > 1.0) {
        return no_hit();
    }
    let t = dot(edge2, q) * inv_det;
    if (t < 0.0 || t > extent) {
        return no_hit();
    }
    let point = origin + direction * t;
    let normal = sign_normalize(cross(edge1, edge2));
    return ShapeHit(t, point, select(normal, -normal, det < 0.0));
}

fn ray_scene(world: WorldShape, origin: vec3f, direction: vec3f, extent: f32) -> ShapeHit {
    let inv_rotation = quat_conjugate(world.rotation);
    let unscale = 1.0 / world.scale;
    let local_origin = (quat_rotate(inv_rotation, origin - world.center)) * unscale;
    let local_direction = (quat_rotate(inv_rotation, direction)) * unscale;
    let local = scene_raycast(world.source, local_origin, local_direction, extent);
    if (local.distance == NO_HIT) {
        return local;
    }
    var hit: ShapeHit;
    hit.distance = local.distance;
    hit.point = world.center + quat_rotate(world.rotation, local.point * world.scale);
    hit.normal = quat_rotate(world.rotation, sign_normalize(local.normal * unscale));
    return hit;
}

fn scene_raycast(source_index: u32, origin: vec3f, direction: vec3f, extent: f32) -> ShapeHit {
    let source = shape_sources[source_index];
    if (source.kind == SHAPE_HULL) {
        var best = no_hit();
        for (var i = 0u; i < source.triangle_count; i = i + 1u) {
            let tri = shape_triangles[source.triangle_offset + i];
            let a = shape_vertices[source.vertex_offset + tri.a].xyz;
            let b = shape_vertices[source.vertex_offset + tri.b].xyz;
            let c = shape_vertices[source.vertex_offset + tri.c].xyz;
            let hit = ray_triangle(origin, direction, extent, a, b, c);
            if (hit.distance < best.distance) {
                best = hit;
            }
        }
        return best;
    }
    var stack: array<u32, 64>;
    var stack_count = 1u;
    stack[0] = source.node_offset;
    var best = no_hit();
    while (stack_count > 0u) {
        stack_count = stack_count - 1u;
        let node_index = stack[stack_count];
        let node = shape_nodes[node_index];
        let t = aabb_ray_hit(node.min, node.max, origin, direction, extent);
        if (t == NO_HIT) {
            continue;
        }
        if (node.leaf == 1u) {
            for (var i = 0u; i < node.right; i = i + 1u) {
                let tri = shape_triangles[source.triangle_offset + node.left + i];
                let a = shape_vertices[source.vertex_offset + tri.a].xyz;
                let b = shape_vertices[source.vertex_offset + tri.b].xyz;
                let c = shape_vertices[source.vertex_offset + tri.c].xyz;
                let hit = ray_triangle(origin, direction, extent, a, b, c);
                if (hit.distance < best.distance) {
                    best = hit;
                }
            }
        } else {
            if (stack_count + 2u > 64u) {
                return best;
            }
            stack[stack_count] = node.left;
            stack_count = stack_count + 1u;
            stack[stack_count] = node.right;
            stack_count = stack_count + 1u;
        }
    }
    return best;
}

fn triangle_points(source_index: u32, triangle_index: u32, scale: vec3f) -> array<vec3f, 3> {
    let source = shape_sources[source_index];
    let tri = shape_triangles[source.triangle_offset + triangle_index];
    let points: array<vec3f, 3> = array(
        shape_vertices[source.vertex_offset + tri.a].xyz * scale,
        shape_vertices[source.vertex_offset + tri.b].xyz * scale,
        shape_vertices[source.vertex_offset + tri.c].xyz * scale,
    );
    return points;
}

fn triangle_world(source_index: u32, triangle_index: u32, scale: vec3f) -> WorldShape {
    let source = shape_sources[source_index];
    let tri = shape_triangles[source.triangle_offset + triangle_index];
    let a = shape_vertices[source.vertex_offset + tri.a].xyz * scale;
    let b = shape_vertices[source.vertex_offset + tri.b].xyz * scale;
    let c = shape_vertices[source.vertex_offset + tri.c].xyz * scale;
    var world: WorldShape;
    world.kind = SHAPE_TRIANGLE;
    world.radius = f32(triangle_index);
    world.half_height = 0.0;
    world.center = (a + b + c) * (1.0 / 3.0);
    world.half_extents = vec3f(0.0);
    world.rotation = vec4f(0.0, 0.0, 0.0, 1.0);
    world.source = source_index;
    world.scale = vec3f(1.0);
    return world;
}

fn triangle_world_points(world: WorldShape, out_points: ptr<function, array<vec3f, 3>>) {
    let source = shape_sources[world.source];
    let tri = shape_triangles[source.triangle_offset + u32(world.radius)];
    (*out_points)[0] = shape_vertices[source.vertex_offset + tri.a].xyz * world.scale;
    (*out_points)[1] = shape_vertices[source.vertex_offset + tri.b].xyz * world.scale;
    (*out_points)[2] = shape_vertices[source.vertex_offset + tri.c].xyz * world.scale;
}

fn sphere_triangle_distance(sphere: WorldShape, points: array<vec3f, 3>) -> ConvexClosest {
    var weights = vec3f(0.0);
    let closest = closest_on_triangle(sphere.center, points[0], points[1], points[2], &weights);
    let delta = closest - sphere.center;
    let distance = length(delta);
    var result: ConvexClosest;
    result.distance = distance - sphere.radius;
    result.point_a = closest;
    result.point_b = sphere.center;
    result.normal = sign_normalize(delta);
    result.penetrating = result.distance <= 0.0;
    return result;
}

fn plane_penetration(triangle: u32, source_index: u32, scale: vec3f, world: WorldShape) -> ConvexClosest {
    let points = triangle_points(source_index, triangle, scale);
    var n = sign_normalize(cross(points[1] - points[0], points[2] - points[0]));
    var offset = dot(world.center - points[0], n);
    if (offset < 0.0) {
        n = -n;
        offset = -offset;
    }
    let deep = support(world, -n);
    let extent = dot(deep - world.center, -n);
    let depth = extent - offset;
    var result: ConvexClosest;
    result.distance = depth;
    result.point_a = world.center - n * extent;
    result.point_b = world.center;
    result.normal = n;
    result.penetrating = depth > 0.0;
    return result;
}

fn scene_convex_closest(source_index: u32, source_scale: vec3f, world: WorldShape, out_triangle: ptr<function, u32>) -> ConvexClosest {
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    var result: ConvexClosest;
    result.distance = 3.402823466e38;
    result.point_a = vec3f(0.0);
    result.point_b = vec3f(0.0);
    result.normal = vec3f(0.0, 1.0, 0.0);
    result.penetrating = false;
    if (shape_sources[source_index].kind == SHAPE_HULL) {
        for (var i = 0u; i < shape_sources[source_index].triangle_count; i = i + 1u) {
            let candidate = convex_closest(triangle_world(source_index, i, source_scale), world, &simplex, &count);
            if (candidate.penetrating) {
                result = candidate;
                result.penetrating = true;
                *out_triangle = i;
                return result;
            }
            if (candidate.distance < result.distance) {
                result = candidate;
                *out_triangle = i;
            }
        }
        return result;
    }
    let world_aabb = world_aabb_of(world);
    let pad = (world_aabb.max - world_aabb.min) * 0.5;
    var stack: array<u32, 64>;
    var stack_count = 1u;
    stack[0] = shape_sources[source_index].node_offset;
    let source = shape_sources[source_index];
    if (source.node_count == 0u) {
        for (var i = 0u; i < source.triangle_count; i = i + 1u) {
            let candidate = plane_penetration(i, source_index, source_scale, world);
            if (candidate.penetrating) { result = candidate; result.penetrating = true; *out_triangle = i; return result; }
            if (candidate.distance < result.distance) { result = candidate; *out_triangle = i; }
        }
        return result;
    }
    while (stack_count > 0u) {
        stack_count = stack_count - 1u;
        let node_index = stack[stack_count];
        let node = shape_nodes[node_index];
        if (node.min.x - pad.x > world_aabb.max.x || node.max.x + pad.x < world_aabb.min.x ||
            node.min.y - pad.y > world_aabb.max.y || node.max.y + pad.y < world_aabb.min.y ||
            node.min.z - pad.z > world_aabb.max.z || node.max.z + pad.z < world_aabb.min.z) {
            continue;
        }
        if (node.leaf == 1u) {
            for (var i = 0u; i < node.right; i = i + 1u) {
                let candidate = plane_penetration(node.left + i, source_index, source_scale, world);
                if (candidate.penetrating) {
                    result = candidate;
                    result.penetrating = true;
                    *out_triangle = node.left + i;
                    return result;
                }
                if (candidate.distance < result.distance) {
                    result = candidate;
                    *out_triangle = node.left + i;
                }
            }
        } else {
            if (stack_count + 2u > 64u) {
                continue;
            }
            stack[stack_count] = node.left;
            stack_count = stack_count + 1u;
            stack[stack_count] = node.right;
            stack_count = stack_count + 1u;
        }
    }
    return result;
}

fn scene_convex_hit(source_index: u32, source_scale: vec3f, world: WorldShape) -> ShapeHit {
    var triangle: u32;
    triangle = 0u;
    let closest = scene_convex_closest(source_index, source_scale, world, &triangle);
    if (!closest.penetrating) {
        if (closest.distance > 0.0) {
            return ShapeHit(closest.distance, (closest.point_a + closest.point_b) * 0.5, closest.normal);
        }
        return no_hit();
    }
    let depth = closest.distance;
    return ShapeHit(-depth, closest.point_a, closest.normal);
}

fn convex_sweep_hit(moving: WorldShape, start: vec3f, direction: vec3f, obstacle: WorldShape) -> ShapeHit {
    var t = 0.0;
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    for (var iter = 0u; iter < 24u; iter = iter + 1u) {
        var moved = moving;
        moved.center = start + direction * t;
        let closest = convex_closest(moved, obstacle, &simplex, &count);
        if (closest.penetrating) {
            return ShapeHit(t, (closest.point_a + closest.point_b) * 0.5, closest.normal);
        }
        if (closest.distance <= 1e-4) {
            return ShapeHit(t, (closest.point_a + closest.point_b) * 0.5, closest.normal);
        }
        t = t + closest.distance;
        if (t > 1e8) {
            return no_hit();
        }
    }
    return no_hit();
}

fn triangle_plane_margin(triangle: u32, source_index: u32, scale: vec3f, world: WorldShape) -> f32 {
    let points = triangle_points(source_index, triangle, scale);
    let n = sign_normalize(cross(points[1] - points[0], points[2] - points[0]));
    let offset = dot(world.center - points[0], n);
    return abs(offset);
}

fn scene_plane_margin(source_index: u32, scale: vec3f, world: WorldShape, out_triangle: ptr<function, u32>) -> f32 {
    let source = shape_sources[source_index];
    let world_aabb = world_aabb_of(world);
    let pad = (world_aabb.max - world_aabb.min) * 0.5;
    var best = 3.402823466e38;
    var stack: array<u32, 64>;
    var stack_count = 1u;
    stack[0] = source.node_offset;
    if (source.node_count == 0u) {
        for (var i = 0u; i < source.triangle_count; i = i + 1u) {
            let margin = triangle_plane_margin(i, source_index, scale, world);
            if (margin < best) {
                best = margin;
                *out_triangle = i;
            }
        }
        return best;
    }
    while (stack_count > 0u) {
        stack_count = stack_count - 1u;
        let node_index = stack[stack_count];
        let node = shape_nodes[node_index];
        if (node.min.x - pad.x > world_aabb.max.x || node.max.x + pad.x < world_aabb.min.x ||
            node.min.y - pad.y > world_aabb.max.y || node.max.y + pad.y < world_aabb.min.y ||
            node.min.z - pad.z > world_aabb.max.z || node.max.z + pad.z < world_aabb.min.z) {
            continue;
        }
        if (node.leaf == 1u) {
            for (var i = 0u; i < node.right; i = i + 1u) {
                let margin = triangle_plane_margin(node.left + i, source_index, scale, world);
                if (margin < best) {
                    best = margin;
                    *out_triangle = node.left + i;
                }
            }
        } else {
            if (stack_count + 2u > 64u) {
                continue;
            }
            stack[stack_count] = node.left;
            stack_count = stack_count + 1u;
            stack[stack_count] = node.right;
            stack_count = stack_count + 1u;
        }
    }
    return best;
}

fn scene_plane_mesh_bounds(source_index: u32, scale: vec3f) -> Aabb {
    let source = shape_sources[source_index];
    if (source.node_count == 0u) {
        var empty: Aabb;
        empty.min = vec3f(3.402823466e38);
        empty.max = vec3f(-3.402823466e38);
        for (var i = 0u; i < source.triangle_count; i = i + 1u) {
            let points = triangle_points(source_index, i, scale);
            empty.min = min(empty.min, points[0]);
            empty.min = min(empty.min, points[1]);
            empty.min = min(empty.min, points[2]);
            empty.max = max(empty.max, points[0]);
            empty.max = max(empty.max, points[1]);
            empty.max = max(empty.max, points[2]);
        }
        return empty;
    }
    let root = shape_nodes[source.node_offset];
    var bounds: Aabb;
    bounds.min = root.min * scale;
    bounds.max = root.max * scale;
    return bounds;
}

fn scene_sweep_hit(
    moving: WorldShape,
    start: vec3f,
    direction: vec3f,
    source_index: u32,
    source_scale: vec3f,
    max_dist: f32,
) -> ShapeHit {
    let moving_bounds = world_aabb_of(moving);
    let probe_radius = length((moving_bounds.max - moving_bounds.min) * 0.5);
    if (probe_radius <= 0.0) {
        return no_hit();
    }
    var triangle = 0u;
    var moved = moving;
    moved.center = start;
    let origin = scene_plane_margin(source_index, source_scale, moved, &triangle);
    if (origin <= probe_radius) {
        return no_hit();
    }
    let bounds = scene_plane_mesh_bounds(source_index, source_scale);
    let entry = ray_box(
        start, direction, max_dist,
        (bounds.min + bounds.max) * 0.5,
        vec4f(0.0, 0.0, 0.0, 1.0),
        (bounds.max - bounds.min) * 0.5 + vec3f(probe_radius),
    );
    if (entry.distance >= max_dist) {
        return no_hit();
    }
    let scan_start = max(entry.distance - probe_radius, 0.0);
    var scan = scan_start;
    var lo = scan_start;
    var hi = -1.0;
    var hit_point = vec3f(0.0);
    var hit_normal = vec3f(0.0, 1.0, 0.0);
    for (var iter = 0u; iter < 4096u; iter = iter + 1u) {
        scan = scan + probe_radius;
        if (scan >= max_dist) {
            return no_hit();
        }
        moved.center = start + direction * scan;
        let margin = scene_plane_margin(source_index, source_scale, moved, &triangle);
        if (margin <= probe_radius) {
            hi = scan;
            break;
        }
        lo = scan;
    }
    if (hi < 0.0) {
        return no_hit();
    }
    for (var iter = 0u; iter < 24u; iter = iter + 1u) {
        let mid = (lo + hi) * 0.5;
        moved.center = start + direction * mid;
        let margin = scene_plane_margin(source_index, source_scale, moved, &triangle);
        if (margin <= probe_radius) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    moved.center = start + direction * (hi + 1e-4);
    let closest = scene_convex_closest(source_index, source_scale, moved, &triangle);
    if (closest.penetrating) {
        return ShapeHit(hi, closest.point_a, closest.normal);
    }
    return ShapeHit(hi, (closest.point_a + closest.point_b) * 0.5, closest.normal);
}

fn plane_normal(world: WorldShape) -> vec3f {
    return quat_rotate(world.rotation, vec3f(0.0, 1.0, 0.0));
}

fn world_rotated_axes(world: WorldShape) -> array<vec3f, 3> {
    let axes: array<vec3f, 3> = array(
        quat_rotate(world.rotation, vec3f(1.0, 0.0, 0.0)),
        quat_rotate(world.rotation, vec3f(0.0, 1.0, 0.0)),
        quat_rotate(world.rotation, vec3f(0.0, 0.0, 1.0)),
    );
    return axes;
}

fn convex_sample_points(world: WorldShape, plane_adverse: vec3f, out_points: ptr<function, array<vec3f, 4>>) -> u32 {
    if (world.kind == SHAPE_SPHERE) {
        (*out_points)[0] = world.center + plane_adverse * world.radius;
        return 1u;
    }
    if (world.kind == SHAPE_CUBOID) {
        let axes = world_rotated_axes(world);
        let e = world.half_extents;
        var count = 0u;
        for (var i = 0u; i < 8u && count < 4u; i = i + 1u) {
            let corner = world.center + axes[0] * select(-e.x, e.x, (i & 1u) != 0u)
                + axes[1] * select(-e.y, e.y, (i & 2u) != 0u)
                + axes[2] * select(-e.z, e.z, (i & 4u) != 0u);
            if (dot(corner - world.center, plane_adverse) < 0.0) {
                (*out_points)[count] = corner;
                count = count + 1u;
            }
        }
        if (count == 0u) {
            (*out_points)[0] = world.center - plane_adverse * min(min(e.x, e.y), e.z);
            return 1u;
        }
        return count;
    }
    if (world.kind == SHAPE_CAPSULE || world.kind == SHAPE_CYLINDER) {
        let axis = shape_axis(world);
        let far = world.center - axis * world.half_height;
        let near = world.center + axis * world.half_height;
        if (dot(far - world.center, plane_adverse) < 0.0) {
            (*out_points)[0] = far;
            (*out_points)[1] = world.center;
            return 2u;
        }
        (*out_points)[0] = near;
        (*out_points)[1] = world.center;
        return 2u;
    }
    let source = shape_sources[world.source];
    var best = vec3f(0.0);
    var best_dot = 3.402823466e38;
    for (var i = 0u; i < source.vertex_count; i = i + 1u) {
        let local = shape_vertices[source.vertex_offset + i].xyz * world.scale;
        let p = world.center + quat_rotate(world.rotation, local);
        let s = dot(p - world.center, plane_adverse);
        if (s < best_dot) {
            best_dot = s;
            best = p;
        }
    }
    (*out_points)[0] = best;
    return 1u;
}

fn ray_scaled_shape(world: WorldShape, origin: vec3f, direction: vec3f, extent: f32, expand: f32) -> ShapeHit {
    let unscaled = world.scale.x == 1.0 && world.scale.y == 1.0 && world.scale.z == 1.0;
    if (unscaled) {
        if (world.kind == SHAPE_SPHERE) {
            return ray_sphere(origin, direction, extent, world.center, world.radius + expand);
        }
        if (world.kind == SHAPE_CUBOID) {
            return ray_box(origin, direction, extent, world.center, world.rotation, world.half_extents + vec3f(expand));
        }
        if (world.kind == SHAPE_CAPSULE) {
            let axis = shape_axis(world);
            return ray_capsule(origin, direction, extent, world.center, axis, world.half_height, world.radius + expand);
        }
        if (world.kind == SHAPE_CYLINDER) {
            let axis = shape_axis(world);
            return ray_cylinder(origin, direction, extent, world.center, axis, world.half_height, world.radius + expand);
        }
        return no_hit();
    }
    let inv_rotation = quat_conjugate(world.rotation);
    let unscale = 1.0 / world.scale;
    let local_origin = quat_rotate(inv_rotation, origin - world.center) * unscale;
    let local_direction = quat_rotate(inv_rotation, direction) * unscale;
    var local: ShapeHit;
    if (world.kind == SHAPE_SPHERE) {
        local = ray_sphere(local_origin, local_direction, extent, vec3f(0.0), world.radius + expand);
    } else if (world.kind == SHAPE_CAPSULE) {
        local = ray_capsule(local_origin, local_direction, extent, vec3f(0.0), vec3f(0.0, 1.0, 0.0), world.half_height, world.radius + expand);
    } else if (world.kind == SHAPE_CYLINDER) {
        local = ray_cylinder(local_origin, local_direction, extent, vec3f(0.0), vec3f(0.0, 1.0, 0.0), world.half_height, world.radius + expand);
    } else {
        return no_hit();
    }
    if (local.distance == NO_HIT) {
        return no_hit();
    }
    var hit: ShapeHit;
    hit.distance = local.distance;
    hit.point = world.center + quat_rotate(world.rotation, (local_origin + local_direction * local.distance) * world.scale);
    hit.normal = quat_rotate(world.rotation, sign_normalize(local.normal * unscale));
    return hit;
}

fn convex_hit_at(
    moving: WorldShape,
    start: vec3f,
    direction: vec3f,
    static_target: WorldShape,
    expand: f32,
    max_dist: f32,
) -> ShapeHit {
    if (static_target.kind == SHAPE_PLANE) {
        let n = plane_normal(static_target);
        let signed = dot(start - static_target.center, n);
        let travel = dot(direction, n);
        if (travel >= 0.0) {
            return no_hit();
        }
        var time = (signed - expand) / travel;
        if (time > max_dist) {
            return no_hit();
        }
        time = max(time, 0.0);
        let point = start + direction * time;
        return ShapeHit(time, point, n);
    }
    if (static_target.kind == SHAPE_HULL) {
        return convex_sweep_hit(moving, start, direction, static_target);
    }
    if (static_target.kind == SHAPE_MESH || static_target.kind == SHAPE_HEIGHTFIELD) {
        return scene_sweep_hit(moving, start, direction, static_target.source, static_target.scale, max_dist);
    }
    return ray_scaled_shape(static_target, start, direction, NO_HIT, expand);
}

fn box_center(state: BodyState, collider: Collider) -> vec3f {
    return state.position + quat_rotate(state.orientation, collider.local_offset);
}

fn box_projected_radius(collider: Collider, q: vec4f, direction: vec3f) -> f32 {
    let local_direction = quat_rotate(quat_conjugate(q), direction);
    return abs(local_direction.x) * collider.half_extents.x
        + abs(local_direction.y) * collider.half_extents.y
        + abs(local_direction.z) * collider.half_extents.z;
}

fn box_face_point(state: BodyState, collider: Collider, direction: vec3f) -> vec3f {
    let q = quat_mul(state.orientation, collider.local_rotation);
    return box_center(state, collider) + direction * box_projected_radius(collider, q, direction);
}

fn box_rotated_axes(state: BodyState, collider: Collider) -> array<vec3f, 3> {
    let q = quat_mul(state.orientation, collider.local_rotation);
    let axes: array<vec3f, 3> = array(
        quat_rotate(q, vec3f(1.0, 0.0, 0.0)),
        quat_rotate(q, vec3f(0.0, 1.0, 0.0)),
        quat_rotate(q, vec3f(0.0, 0.0, 1.0)),
    );
    return axes;
}

fn closest_point_box(point: vec3f, state: BodyState, collider: Collider) -> vec3f {
    let q = quat_mul(state.orientation, collider.local_rotation);
    let local = quat_rotate(quat_conjugate(q), point - box_center(state, collider));
    let clamped = clamp(local, -collider.half_extents, collider.half_extents);
    return box_center(state, collider) + quat_rotate(q, clamped);
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

const FEATURE_MAX: u32 = 8u;

fn manifold_push(contact: ptr<function, Contact>, point: vec3f, depth: f32) {
    let count = (*contact).point_count;
    if (count >= CONTACT_MAX_POINTS) {
        return;
    }
    (*contact).points[count] = ManifoldPoint(point, depth, 0.0, 0.0, 0.0, 0.0);
    (*contact).point_count = count + 1u;
}

fn box_feature_ring(world: WorldShape, direction: vec3f, out: ptr<function, array<vec3f, FEATURE_MAX>>) -> u32 {
    let axes = world_rotated_axes(world);
    var best_axis = 0u;
    for (var i = 1u; i < 3u; i = i + 1u) {
        if (abs(dot(axes[i], direction)) > abs(dot(axes[best_axis], direction))) {
            best_axis = i;
        }
    }
    let sign = select(1.0, -1.0, dot(axes[best_axis], direction) < 0.0);
    let face = axes[best_axis] * sign;
    let center = world.center + face * world.half_extents[best_axis];
    let u1 = axes[(best_axis + 1u) % 3u];
    let u2 = axes[(best_axis + 2u) % 3u];
    let e1 = world.half_extents[(best_axis + 1u) % 3u];
    let e2 = world.half_extents[(best_axis + 2u) % 3u];
    (*out)[0] = center + u1 * e1 + u2 * e2;
    (*out)[1] = center - u1 * e1 + u2 * e2;
    (*out)[2] = center - u1 * e1 - u2 * e2;
    (*out)[3] = center + u1 * e1 - u2 * e2;
    return 4u;
}

fn hull_feature_ring(world: WorldShape, direction: vec3f, out: ptr<function, array<vec3f, FEATURE_MAX>>) -> u32 {
    let source = shape_sources[world.source];
    var best_dot = -3.402823466e38;
    for (var i = 0u; i < source.vertex_count; i = i + 1u) {
        let local = shape_vertices[source.vertex_offset + i].xyz * world.scale;
        let p = world.center + quat_rotate(world.rotation, local);
        let d = dot(p, direction);
        if (d > best_dot) {
            best_dot = d;
        }
    }
    var points: array<vec3f, 16>;
    var count = 0u;
    let eps = 1e-3 * shape_scale(world);
    for (var i = 0u; i < source.vertex_count && count < 16u; i = i + 1u) {
        let local = shape_vertices[source.vertex_offset + i].xyz * world.scale;
        let p = world.center + quat_rotate(world.rotation, local);
        if (dot(p, direction) >= best_dot - eps) {
            points[count] = p;
            count = count + 1u;
        }
    }
    if (count <= 1u) {
        (*out)[0] = support(world, direction);
        return 1u;
    }
    let n = sign_normalize(direction);
    var center = vec3f(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        center = center + points[i];
    }
    center = center / f32(count);
    var u = vec3f(0.0);
    if (abs(n.x) > 0.9) {
        u = normalize(cross(n, vec3f(0.0, 1.0, 0.0)));
    } else {
        u = normalize(cross(n, vec3f(1.0, 0.0, 0.0)));
    }
    let v = cross(n, u);
    for (var i = 0u; i < count; i = i + 1u) {
        for (var j = i + 1u; j < count; j = j + 1u) {
            let pa = points[i] - center;
            let pb = points[j] - center;
            let angle_a = atan2(dot(pa, v), dot(pa, u));
            let angle_b = atan2(dot(pb, v), dot(pb, u));
            if (angle_b < angle_a) {
                let tmp = points[i];
                points[i] = points[j];
                points[j] = tmp;
            }
        }
    }
    var keep = min(count, FEATURE_MAX);
    for (var i = 0u; i < keep; i = i + 1u) {
        (*out)[i] = points[i];
    }
    return keep;
}

fn cylinder_feature(world: WorldShape, direction: vec3f, out: ptr<function, array<vec3f, FEATURE_MAX>>) -> u32 {
    let axis = shape_axis(world);
    let alignment = dot(axis, normalize(direction));
    if (abs(alignment) > 0.93) {
        let sign = select(1.0, -1.0, alignment < 0.0);
        let cap_center = world.center + axis * world.half_height * sign;
        var u = normalize(cross(axis, vec3f(1.0, 0.0, 0.0)));
        if (length(cross(axis, vec3f(1.0, 0.0, 0.0))) < 1e-6) {
            u = normalize(cross(axis, vec3f(0.0, 0.0, 1.0)));
        }
        let v = cross(axis, u);
        for (var i = 0u; i < 8u; i = i + 1u) {
            let angle = f32(i) * 0.78539816;
            (*out)[i] = cap_center + (u * cos(angle) + v * sin(angle)) * world.radius;
        }
        return 8u;
    }
    let side = sign_normalize(direction - axis * dot(direction, axis));
    (*out)[0] = world.center + axis * world.half_height + side * world.radius;
    (*out)[1] = world.center - axis * world.half_height + side * world.radius;
    return 2u;
}

fn shape_feature(
    world: WorldShape,
    direction: vec3f,
    out: ptr<function, array<vec3f, FEATURE_MAX>>,
    flat: ptr<function, bool>,
) -> u32 {
    if (world.kind == SHAPE_CUBOID) {
        *flat = true;
        return box_feature_ring(world, direction, out);
    }
    if (world.kind == SHAPE_HULL) {
        let count = hull_feature_ring(world, direction, out);
        *flat = count > 1u;
        return count;
    }
    if (world.kind == SHAPE_CYLINDER) {
        let count = cylinder_feature(world, direction, out);
        *flat = count > 2u;
        return count;
    }
    *flat = false;
    (*out)[0] = support(world, direction);
    return 1u;
}

fn convex_pair_manifold(
    first: WorldShape,
    second: WorldShape,
    direction: vec3f,
    contact: ptr<function, Contact>,
) -> bool {
    var first_points: array<vec3f, FEATURE_MAX>;
    var second_points: array<vec3f, FEATURE_MAX>;
    var first_flat = false;
    var second_flat = false;
    let first_count = shape_feature(first, direction, &first_points, &first_flat);
    let second_count = shape_feature(second, -direction, &second_points, &second_flat);
    if (!first_flat && !second_flat) {
        return false;
    }
    var reference: array<vec3f, FEATURE_MAX>;
    var reference_count = 0u;
    var incident: array<vec3f, FEATURE_MAX>;
    var incident_count = 0u;
    var ref_dir: vec3f;
    if (first_flat) {
        reference = first_points;
        reference_count = first_count;
        incident = second_points;
        incident_count = second_count;
        ref_dir = direction;
    } else {
        reference = second_points;
        reference_count = second_count;
        incident = first_points;
        incident_count = first_count;
        ref_dir = -direction;
    }
    var center = vec3f(0.0);
    for (var i = 0u; i < reference_count; i = i + 1u) {
        center = center + reference[i];
    }
    center = center / f32(reference_count);
    var candidates: array<ManifoldPoint, 4>;
    var candidate_count = 0u;
    for (var i = 0u; i < incident_count && candidate_count < 4u; i = i + 1u) {
        let point = incident[i];
        let depth = dot(center - point, ref_dir);
        if (depth > 0.0) {
            var kept = true;
            for (var s = 0u; s < reference_count && kept; s = s + 1u) {
                let edge = reference[(s + 1u) % reference_count] - reference[s];
                let plane_normal = normalize(cross(edge, ref_dir));
                if (dot(point - reference[s], plane_normal) < 0.0) {
                    kept = false;
                }
            }
            if (kept) {
                candidates[candidate_count] = ManifoldPoint(point, depth, 0.0, 0.0, 0.0, 0.0);
                candidate_count = candidate_count + 1u;
            }
        }
    }
    if (candidate_count == 0u) {
        return false;
    }
    for (var i = 0u; i < candidate_count; i = i + 1u) {
        for (var j = i + 1u; j < candidate_count; j = j + 1u) {
            if (candidates[j].depth > candidates[i].depth) {
                let tmp = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = tmp;
            }
        }
    }
    for (var i = 0u; i < candidate_count; i = i + 1u) {
        manifold_push(contact, candidates[i].position, candidates[i].depth);
    }
    return true;
}

fn scene_convex_manifold(
    source_index: u32,
    source_scale: vec3f,
    world: WorldShape,
    contact: ptr<function, Contact>,
) -> bool {
    let source = shape_sources[source_index];
    var candidates: array<ManifoldPoint, 8>;
    var candidate_count = 0u;
    if (source.node_count == 0u) {
        for (var i = 0u; i < source.triangle_count; i = i + 1u) {
            let probe = plane_penetration(i, source_index, source_scale, world);
            if (probe.penetrating) {
                candidates[candidate_count] = ManifoldPoint(probe.point_a, probe.distance, 0.0, 0.0, 0.0, 0.0);
                candidate_count = candidate_count + 1u;
                if (candidate_count >= 8u) {
                    break;
                }
            }
        }
    } else {
        var stack: array<u32, 64>;
        var stack_count = 1u;
        stack[0] = source.node_offset;
        let world_aabb = world_aabb_of(world);
        let pad = (world_aabb.max - world_aabb.min) * 0.5;
        while (stack_count > 0u && candidate_count < 8u) {
            stack_count = stack_count - 1u;
            let node_index = stack[stack_count];
            let node = shape_nodes[node_index];
            if (node.min.x - pad.x > world_aabb.max.x || node.max.x + pad.x < world_aabb.min.x ||
                node.min.y - pad.y > world_aabb.max.y || node.max.y + pad.y < world_aabb.min.y ||
                node.min.z - pad.z > world_aabb.max.z || node.max.z + pad.z < world_aabb.min.z) {
                continue;
            }
            if (node.leaf == 1u) {
                for (var i = 0u; i < node.right && candidate_count < 8u; i = i + 1u) {
                    let probe = plane_penetration(node.left + i, source_index, source_scale, world);
                    if (probe.penetrating) {
                        candidates[candidate_count] = ManifoldPoint(probe.point_a, probe.distance, 0.0, 0.0, 0.0, 0.0);
                        candidate_count = candidate_count + 1u;
                    }
                }
            } else {
                if (stack_count + 2u > 64u) {
                    continue;
                }
                stack[stack_count] = node.left;
                stack_count = stack_count + 1u;
                stack[stack_count] = node.right;
                stack_count = stack_count + 1u;
            }
        }
    }
    if (candidate_count == 0u) {
        return false;
    }
    var keep = min(candidate_count, CONTACT_MAX_POINTS);
    for (var i = 0u; i < candidate_count; i = i + 1u) {
        for (var j = i + 1u; j < candidate_count; j = j + 1u) {
            if (candidates[j].depth > candidates[i].depth) {
                let tmp = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = tmp;
            }
        }
    }
    for (var i = 0u; i < keep; i = i + 1u) {
        let point = candidates[i].position;
        var merged = false;
        for (var j = 0u; j < i; j = j + 1u) {
            if (length(point - candidates[j].position) < 0.05 * shape_scale(world)) {
                merged = true;
            }
        }
        if (!merged) {
            manifold_push(contact, point, candidates[i].depth);
        }
    }
    return contact.point_count > 0u;
}
