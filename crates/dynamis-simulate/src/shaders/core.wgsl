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
    edit_run_count: u32,
    body_move_count: u32,
    constraint_move_count: u32,
    event_slot: u32,
    _pad0: u32,
    _pad1: u32,
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

struct Body {
    state: BodyState,
    desc: BodyDescriptor,
}

struct RowMove {
    row: u32,
    source: u32,
    fresh: u32,
    _pad: u32,
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

struct TangentBasis {
    first: vec3f,
    second: vec3f,
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

struct ConvexClosest {
    distance: f32,
    point_a: vec3f,
    point_b: vec3f,
    normal: vec3f,
    penetrating: bool,
}

struct ShapeHit {
    distance: f32,
    point: vec3f,
    normal: vec3f,
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

fn body_is_movable(desc: BodyDescriptor) -> bool {
    return desc.inverse_mass > 0.0 || (desc.flags & BODY_KINEMATIC) != 0u;
}

fn body_is_active(state: BodyState, desc: BodyDescriptor) -> bool {
    return body_is_movable(desc) && state.sleeping == 0u;
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

