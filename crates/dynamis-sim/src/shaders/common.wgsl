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
    _pad0: u32,
    sleep_velocity: f32,
    sleep_angular_velocity: f32,
    sleep_time: f32,
    wake_velocity: f32,
    _pad5: f32,
}

struct RigidBody {
    position: vec3f,
    _pad0: f32,
    prev_position: vec3f,
    _prev_pad: f32,
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
    collider_count: u32,
    flags: u32,
    collision_group: u32,
    collision_mask: u32,
    sleep_timer: f32,
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
    kind: u32,
    flags: u32,
    radius: f32,
    half_height: f32,
    half_extents: vec3f,
    _pad0: f32,
    local_offset: vec3f,
    _pad1: f32,
    local_rotation: vec4f,
    friction: f32,
    restitution: f32,
    source: u32,
    _pad2: u32,
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
    _pad2: f32,
    points: array<ManifoldPoint, CONTACT_MAX_POINTS>,
}

struct Constraint {
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
    motor_speed: f32,
    spring_frequency: f32,
    spring_damping_ratio: f32,
    _pad5: f32,
    _pad6: f32,
    accumulated: array<f32, 8>,
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
    origin: vec3f,
    _pad0: f32,
    direction: vec3f,
    extent: f32,
    radius: f32,
    half_height: f32,
    _pad1: f32,
    _pad1b: f32,
    half_extents: vec3f,
    _pad2: f32,
    orientation: vec4f,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
    _pad6: f32,
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
    _pad0: u32,
    point: vec3f,
    _pad1: f32,
    normal: vec3f,
    _pad2: f32,
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

fn body_is_inert(body: RigidBody) -> bool {
    return body.inverse_mass == 0.0 || (body.flags & BODY_SLEEPING) != 0u;
}

fn body_frozen(body: RigidBody) -> RigidBody {
    var frozen = body;
    frozen.inverse_mass = 0.0;
    frozen.inverse_inertia_body = vec3f(0.0);
    return frozen;
}

fn body_is_dynamic(body: RigidBody) -> bool {
    return body.inverse_mass > 0.0 && (body.flags & BODY_KINEMATIC) == 0u;
}

fn body_has_ccd(body: RigidBody) -> bool {
    return (body.flags & BODY_CCD) != 0u;
}

fn body_is_static(body: RigidBody) -> bool {
    return body.inverse_mass == 0.0 && (body.flags & BODY_KINEMATIC) == 0u;
}

fn collider_is_sensor(collider: Collider) -> bool {
    return (collider.flags & COLLIDER_SENSOR) != 0u;
}

fn body_world_intersects(body: RigidBody, group: u32, mask: u32) -> bool {
    return (body.collision_group & mask) != 0u && (group & body.collision_mask) != 0u;
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

struct WorldShape {
    kind: u32,
    radius: f32,
    half_height: f32,
    center: vec3f,
    half_extents: vec3f,
    rotation: vec4f,
    source: u32,
}

fn world_collider(body: RigidBody, collider: Collider) -> WorldShape {
    var world: WorldShape;
    world.kind = collider.kind;
    world.radius = collider.radius;
    world.half_height = collider.half_height;
    world.center = body.position + quat_rotate(body.orientation, collider.local_offset);
    world.half_extents = collider.half_extents;
    world.rotation = quat_mul(body.orientation, collider.local_rotation);
    world.source = collider.source;
    return world;
}

fn shape_axis(world: WorldShape) -> vec3f {
    return quat_rotate(world.rotation, vec3f(0.0, 1.0, 0.0));
}

fn support(world: WorldShape, direction: vec3f) -> vec3f {
    let d = direction;
    let n = normalize(direction);
    if (world.kind == SHAPE_SPHERE) {
        return world.center + n * world.radius;
    }
    if (world.kind == SHAPE_BOX) {
        let local = quat_rotate(quat_conjugate(world.rotation), d);
        let signs = select(vec3f(-1.0), vec3f(1.0), local > vec3f(0.0));
        return world.center + quat_rotate(world.rotation, signs * world.half_extents);
    }
    if (world.kind == SHAPE_CAPSULE) {
        let axis = shape_axis(world);
        let tip = world.center + axis * world.half_height * select(-1.0, 1.0, dot(d, axis) > 0.0);
        return tip + n * world.radius;
    }
    if (world.kind == SHAPE_CYLINDER) {
        let axis = shape_axis(world);
        let along = dot(d, axis);
        let base = world.center + axis * world.half_height * select(-1.0, 1.0, along > 0.0);
        let radial = d - axis * along;
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
    let local_d = quat_rotate(quat_conjugate(world.rotation), d);
    var best_dot = -3.402823466e38;
    var best = world.center;
    for (var i = 0u; i < source.vertex_count; i = i + 1u) {
        let v = shape_vertices[source.vertex_offset + i].xyz;
        let s = dot(v, local_d);
        if (s > best_dot) {
            best_dot = s;
            best = world.center + quat_rotate(world.rotation, v);
        }
    }
    return best;
}

fn triangle_vertex(world: WorldShape, index: u32) -> vec3f {
    let source = shape_sources[world.source];
    let local = shape_vertices[source.vertex_offset + index].xyz;
    return world.center + quat_rotate(world.rotation, local);
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
    return max(scale, 1e-4);
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
            return ShapeHit(-probe.depth, (closest.point_a + closest.point_b) * 0.5, probe.normal);
        }
        if (closest.distance > 0.0) {
            return ShapeHit(closest.distance, (closest.point_a + closest.point_b) * 0.5, closest.normal);
        }
        return no_hit();
    }
    let epa = epa_tetrahedron(first, second, simplex, count);
    var result: ShapeHit;
    if (epa.valid && epa.depth > 0.0 && epa.depth < 3.402823466e38 && length(epa.normal) > 0.5 && epa.depth < support_projection_depth(first, second, epa.normal) + 0.1 * min(shape_scale(first), shape_scale(second))) {
        result = ShapeHit(-epa.depth, (closest.point_a + closest.point_b) * 0.5, epa.normal);
    } else {
        let probe = convex_penetration_probe(first, second, simplex, count);
        result = ShapeHit(-probe.depth, (closest.point_a + closest.point_b) * 0.5, probe.normal);
    }
    return result;
}

fn min_radius(collider: Collider) -> f32 {
    if (collider.kind == SHAPE_SPHERE) {
        return collider.radius;
    }
    if (collider.kind == SHAPE_BOX) {
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
    var extent = vec3f(0.0);
    if (world.kind == SHAPE_SPHERE) {
        extent = vec3f(world.radius);
    } else if (world.kind == SHAPE_BOX) {
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
            let p = triangle_vertex(world, i);
            min_v = min(min_v, p);
            max_v = max(max_v, p);
        }
        extent = (max_v - min_v) * 0.5;
    }
    var aabb: Aabb;
    aabb.min = world.center - extent;
    aabb.max = world.center + extent;
    return aabb;
}

fn aabb_overlap(first: Aabb, second: Aabb) -> bool {
    return all(first.min <= second.max) && all(second.min <= first.max);
}

fn collider_world_aabb(body: RigidBody, collider: Collider) -> Aabb {
    let world = world_collider(body, collider);
    return world_aabb_of(world);
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

fn scene_raycast(source_index: u32, origin: vec3f, direction: vec3f, extent: f32) -> ShapeHit {
    let source = shape_sources[source_index];
    if (source.kind == SHAPE_SOURCE_HULL) {
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

fn triangle_points(source_index: u32, triangle_index: u32) -> array<vec3f, 3> {
    let source = shape_sources[source_index];
    let tri = shape_triangles[source.triangle_offset + triangle_index];
    let points: array<vec3f, 3> = array(
        shape_vertices[source.vertex_offset + tri.a].xyz,
        shape_vertices[source.vertex_offset + tri.b].xyz,
        shape_vertices[source.vertex_offset + tri.c].xyz,
    );
    return points;
}

fn triangle_world(source_index: u32, triangle_index: u32) -> WorldShape {
    let source = shape_sources[source_index];
    let tri = shape_triangles[source.triangle_offset + triangle_index];
    let a = shape_vertices[source.vertex_offset + tri.a].xyz;
    let b = shape_vertices[source.vertex_offset + tri.b].xyz;
    let c = shape_vertices[source.vertex_offset + tri.c].xyz;
    var world: WorldShape;
    world.kind = SHAPE_TRIANGLE;
    world.radius = f32(triangle_index);
    world.half_height = 0.0;
    world.center = (a + b + c) * (1.0 / 3.0);
    world.half_extents = vec3f(0.0);
    world.rotation = vec4f(0.0, 0.0, 0.0, 1.0);
    world.source = source_index;
    return world;
}

fn triangle_world_points(world: WorldShape, out_points: ptr<function, array<vec3f, 3>>) {
    let source = shape_sources[world.source];
    let tri = shape_triangles[source.triangle_offset + u32(world.radius)];
    (*out_points)[0] = shape_vertices[source.vertex_offset + tri.a].xyz;
    (*out_points)[1] = shape_vertices[source.vertex_offset + tri.b].xyz;
    (*out_points)[2] = shape_vertices[source.vertex_offset + tri.c].xyz;
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

fn plane_penetration(triangle: u32, source_index: u32, world: WorldShape) -> ConvexClosest {
    let points = triangle_points(source_index, triangle);
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

fn scene_convex_closest(source_index: u32, world: WorldShape, out_triangle: ptr<function, u32>) -> ConvexClosest {
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    var result: ConvexClosest;
    result.distance = 3.402823466e38;
    result.point_a = vec3f(0.0);
    result.point_b = vec3f(0.0);
    result.normal = vec3f(0.0, 1.0, 0.0);
    result.penetrating = false;
    if (shape_sources[source_index].kind == SHAPE_SOURCE_HULL) {
        for (var i = 0u; i < shape_sources[source_index].triangle_count; i = i + 1u) {
            let candidate = convex_closest(triangle_world(source_index, i), world, &simplex, &count);
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
            let candidate = plane_penetration(i, source_index, world);
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
                let candidate = plane_penetration(node.left + i, source_index, world);
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

fn scene_convex_hit(source_index: u32, world: WorldShape) -> ShapeHit {
    var triangle: u32;
    triangle = 0u;
    let closest = scene_convex_closest(source_index, world, &triangle);
    if (!closest.penetrating) {
        if (closest.distance > 0.0) {
            return ShapeHit(closest.distance, (closest.point_a + closest.point_b) * 0.5, closest.normal);
        }
        return no_hit();
    }
    let depth = closest.distance;
    return ShapeHit(-depth, closest.point_a, closest.normal);
}

fn convex_hit_at(
    moving: WorldShape,
    start: vec3f,
    direction: vec3f,
    static_target: WorldShape,
    expand: f32,
) -> ShapeHit {
    if (static_target.kind == SHAPE_SPHERE) {
        return ray_sphere(start, direction, NO_HIT, static_target.center, static_target.radius + expand);
    }
    if (static_target.kind == SHAPE_BOX) {
        let q = static_target.rotation;
        return ray_box(start, direction, NO_HIT, static_target.center, q, static_target.half_extents + vec3f(expand));
    }
    if (static_target.kind == SHAPE_CAPSULE) {
        let axis = shape_axis(static_target);
        return ray_capsule(start, direction, NO_HIT, static_target.center, axis, static_target.half_height, static_target.radius + expand);
    }
    if (static_target.kind == SHAPE_CYLINDER) {
        let axis = shape_axis(static_target);
        return ray_cylinder(start, direction, NO_HIT, static_target.center, axis, static_target.half_height, static_target.radius + expand);
    }
    return scene_raycast(static_target.source, start, direction, NO_HIT);
}

fn box_center(body: RigidBody, collider: Collider) -> vec3f {
    return body.position + quat_rotate(body.orientation, collider.local_offset);
}

fn box_projected_radius(collider: Collider, q: vec4f, direction: vec3f) -> f32 {
    let local_direction = quat_rotate(quat_conjugate(q), direction);
    return abs(local_direction.x) * collider.half_extents.x
        + abs(local_direction.y) * collider.half_extents.y
        + abs(local_direction.z) * collider.half_extents.z;
}

fn box_face_point(body: RigidBody, collider: Collider, direction: vec3f) -> vec3f {
    let q = quat_mul(body.orientation, collider.local_rotation);
    return box_center(body, collider) + direction * box_projected_radius(collider, q, direction);
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
