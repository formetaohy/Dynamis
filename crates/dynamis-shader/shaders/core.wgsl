struct Body {
    state: BodyState,
    desc: BodyDescriptor,
}

const SOLVER_VELOCITY_SCALE: f32 = 262144.0;
const SOLVER_POSITION_SCALE: f32 = 16777216.0;

fn solver_word(value: f32, scale: f32) -> u32 {
    return u32(i32(clamp(value * scale, -2.0e9, 2.0e9)));
}

fn solver_value(word: u32, scale: f32) -> f32 {
    return f32(i32(word)) / scale;
}

const REACTION_SCALE: f32 = 65536.0;

fn reaction_word(value: f32) -> u32 {
    return u32(i32(clamp(value * REACTION_SCALE, -2.0e9, 2.0e9)));
}

fn reaction_value(word: u32) -> f32 {
    return f32(i32(word)) / REACTION_SCALE;
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
    triangle: u32,
}

fn global_index(gid: vec3u) -> u32 {
    return gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
}

fn grid_stride(groups: vec3u) -> u32 {
    return groups.x * groups.y * WORKGROUP_SIZE;
}

const GRID_SCALE_LIMIT: f32 = 1e12;
const GRID_LEVEL_LIMIT: u32 = 30u;

fn grid_scale_of(bounds: Aabb) -> f32 {
    let extent = grid_extent_of(bounds);
    if (extent <= 0.0) {
        return 0.0;
    }
    return clamp(1.0 / extent, 1.0 / GRID_SCALE_LIMIT, GRID_SCALE_LIMIT);
}

fn grid_extent_of(bounds: Aabb) -> f32 {
    return max(max(bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y), bounds.max.z - bounds.min.z);
}

fn grid_cell_size(scale_bits: u32, extent_bits: u32) -> f32 {
    let scale = bitcast<f32>(scale_bits);
    let extent = bitcast<f32>(extent_bits);
    let finest = select(GRID_SCALE_LIMIT, 1.0 / scale, scale > 0.0);
    return max(finest, extent * exp2(-f32(GRID_LEVEL_LIMIT)));
}

fn aabb_overlaps(first: Aabb, second: Aabb) -> bool {
    return all(first.min <= second.max) && all(second.min <= first.max);
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

fn rotate_about(rotation: vec3f, v: vec3f) -> vec3f {
    let angle = length(rotation);
    if (angle < 1e-8) {
        return v;
    }
    let axis = rotation / angle;
    let offset = cross(axis, v);
    return v + sin(angle) * offset + (1.0 - cos(angle)) * cross(axis, offset);
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

fn freeze_body(state: ptr<function, BodyState>) {
    (*state).velocity = vec3f(0.0);
    (*state).angular_velocity = vec3f(0.0);
    (*state).force = vec3f(0.0);
    (*state).torque = vec3f(0.0);
    (*state).prev_position = (*state).position;
    (*state).sleep_timer = 0.0;
    (*state).sleeping = 1u;
}

fn body_is_movable(desc: BodyDescriptor) -> bool {
    return desc.inverse_mass > 0.0 || (desc.flags & BODY_KINEMATIC) != 0u;
}

fn body_is_active(state: BodyState, desc: BodyDescriptor) -> bool {
    return body_is_movable(desc) && state.sleeping == 0u;
}

fn body_sleep_velocity(desc: BodyDescriptor, params: StepParams) -> f32 {
    return select(params.sleep_velocity, desc.sleep_velocity, (desc.flags & OVERRIDE_SLEEP_LINEAR) != 0u);
}

fn body_sleep_angular_velocity(desc: BodyDescriptor, params: StepParams) -> f32 {
    return select(params.sleep_angular_velocity, desc.sleep_angular_velocity, (desc.flags & OVERRIDE_SLEEP_ANGULAR) != 0u);
}

fn body_is_driven_in_motion(state: BodyState, desc: BodyDescriptor) -> bool {
    return (desc.flags & BODY_KINEMATIC) != 0u
        && (any(state.velocity != vec3f(0.0)) || any(state.angular_velocity != vec3f(0.0)));
}

fn body_is_moving(state: BodyState, desc: BodyDescriptor, params: StepParams) -> bool {
    if (body_is_driven_in_motion(state, desc)) {
        return true;
    }
    if (desc.inverse_mass == 0.0 || state.sleeping != 0u) {
        return false;
    }
    return length(state.velocity) > body_sleep_velocity(desc, params)
        || length(state.angular_velocity) > body_sleep_angular_velocity(desc, params);
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

fn filters_intersect(first: vec2u, second: vec2u) -> bool {
    return (first.x & second.y) != 0u && (second.x & first.y) != 0u;
}

fn collider_filter(body: Body, collider: Collider) -> vec2u {
    let group = select(body.desc.collision_group, collider.collision_group, collider.collision_group != NO_COLLISION_FILTER);
    let mask = select(body.desc.collision_mask, collider.collision_mask, collider.collision_mask != NO_COLLISION_FILTER);
    return vec2u(group, mask);
}

fn collider_filter_intersects(
    first_body: Body, first_collider: Collider,
    second_body: Body, second_collider: Collider,
) -> bool {
    return filters_intersect(
        collider_filter(first_body, first_collider),
        collider_filter(second_body, second_collider),
    );
}

fn collider_filter_query(query: Query, body: Body, collider: Collider) -> bool {
    if (query.group == 0u) {
        return true;
    }
    return filters_intersect(vec2u(query.group, query.mask), collider_filter(body, collider));
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

fn collider_surface(collider: Collider) -> Surface {
    return Surface(collider.friction, collider.restitution, collider.rolling_friction, collider.spin_friction);
}

fn body_com(body: Body) -> vec3f {
    return body_com_of(body.state, body.desc);
}

fn body_com_of(state: BodyState, desc: BodyDescriptor) -> vec3f {
    return state.position + quat_rotate(state.orientation, desc.com);
}

fn symmetric_apply(tensor: array<f32, 6>, v: vec3f) -> vec3f {
    return vec3f(
        tensor[0] * v.x + tensor[1] * v.y + tensor[2] * v.z,
        tensor[1] * v.x + tensor[3] * v.y + tensor[4] * v.z,
        tensor[2] * v.x + tensor[4] * v.y + tensor[5] * v.z,
    );
}

fn inertia_local(desc: BodyDescriptor, v: vec3f) -> vec3f {
    return symmetric_apply(desc.inertia, v);
}

fn inertia_is_isotropic(desc: BodyDescriptor) -> bool {
    return desc.inertia[1] == 0.0
        && desc.inertia[2] == 0.0
        && desc.inertia[4] == 0.0
        && desc.inertia[0] == desc.inertia[3]
        && desc.inertia[3] == desc.inertia[5];
}

fn inverse_inertia_local(desc: BodyDescriptor, v: vec3f) -> vec3f {
    return symmetric_apply(desc.inverse_inertia, v);
}

fn apply_inverse_inertia_of(desc: BodyDescriptor, q: vec4f, v: vec3f) -> vec3f {
    let local = quat_rotate(quat_conjugate(q), v);
    return quat_rotate(q, inverse_inertia_local(desc, local));
}

fn apply_inverse_inertia(body: Body, v: vec3f) -> vec3f {
    return apply_inverse_inertia_of(body.desc, body.state.orientation, v);
}

fn point_velocity(body: Body, point: vec3f) -> vec3f {
    return body.state.velocity + cross(body.state.angular_velocity, point - body_com(body));
}

fn relative_velocity(body_a: Body, body_b: Body, point_a: vec3f, point_b: vec3f) -> vec3f {
    return point_velocity(body_b, point_b) - point_velocity(body_a, point_a);
}

fn orthogonal_axis(index: u32) -> vec3f {
    if (index == 0u) {
        return vec3f(1.0, 0.0, 0.0);
    }
    if (index == 1u) {
        return vec3f(0.0, 1.0, 0.0);
    }
    return vec3f(0.0, 0.0, 1.0);
}

fn constraint_anchor(body: Body, local: vec3f) -> vec3f {
    return body.state.position + quat_rotate(body.state.orientation, local);
}

fn constraint_local_frame(local_axis: vec3f) -> TangentBasis {
    return make_tangents(normalize(local_axis));
}

fn constraint_dof_axis(tangents: TangentBasis, hinge: vec3f, index: u32) -> vec3f {
    if (index == 0u) {
        return tangents.first;
    }
    if (index == 1u) {
        return tangents.second;
    }
    return hinge;
}

fn vec_index(v: vec3f, index: u32) -> f32 {
    return select(select(v.x, v.z, index == 2u), v.y, index == 1u);
}

fn constraint_relative_error(first: Body, second: Body, reference: vec4f) -> vec3f {
    let q_rel = quat_mul(quat_conjugate(first.state.orientation), second.state.orientation);
    let deviation = quat_mul(q_rel, quat_conjugate(reference));
    return 2.0 * deviation.xyz;
}

fn constraint_angle(first: Body, second: Body, local_hinge: vec3f) -> f32 {
    let q_rel = quat_mul(quat_conjugate(first.state.orientation), second.state.orientation);
    return 2.0 * atan2(dot(q_rel.xyz, local_hinge), q_rel.w);
}

fn contact_block_resolves(contact: Contact) -> bool {
    return contact.point_count > 0u && contact.sensor == 0u;
}

fn linear_momentum_mass(body_a: Body, body_b: Body) -> f32 {
    return body_a.desc.inverse_mass + body_b.desc.inverse_mass;
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

fn world_shape_bounds(world: WorldShape) -> Aabb {
    let source = shape_sources[world.source];
    let local_center = (source.local_min + source.local_max) * 0.5;
    let half = (source.local_max - source.local_min) * 0.5;
    let center = world.center + quat_rotate(world.rotation, local_center);
    let extent = abs(quat_rotate(world.rotation, vec3f(half.x, 0.0, 0.0)))
        + abs(quat_rotate(world.rotation, vec3f(0.0, half.y, 0.0)))
        + abs(quat_rotate(world.rotation, vec3f(0.0, 0.0, half.z)));
    var aabb: Aabb;
    aabb.min = center - extent;
    aabb.max = center + extent;
    return aabb;
}

fn world_aabb_of(world: WorldShape) -> Aabb {
    if (world.kind == SHAPE_PLANE) {
        var plane_box: Aabb;
        plane_box.min = world.center - vec3f(1e6);
        plane_box.max = world.center + vec3f(1e6);
        return plane_box;
    }
    if (world.kind == SHAPE_HULL || world.kind == SHAPE_MESH || world.kind == SHAPE_HEIGHTFIELD) {
        let bounds = world_shape_bounds(world);
        let s = max(max(world.scale.x, world.scale.y), world.scale.z);
        let center = (bounds.min + bounds.max) * 0.5;
        let extent = (bounds.max - bounds.min) * 0.5 * s;
        var aabb: Aabb;
        aabb.min = center - extent;
        aabb.max = center + extent;
        return aabb;
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

fn smallest_axis(v: vec3f) -> u32 {
    let av = abs(v);
    if (av.x < av.y && av.x < av.z) {
        return 0u;
    }
    if (av.y < av.z) {
        return 1u;
    }
    return 2u;
}

const FEATURE_MAX: u32 = 8u;
const CLIP_MARGIN: f32 = 1e-4;
const LINEAR_SUPPORT_VERTICES: u32 = 4096u;

fn feature_point() -> u32 {
    return FEATURE_POINT;
}

fn feature_pair(kind: u32, first: u32, second: u32) -> u32 {
    return kind | (first << FEATURE_FIELD_BITS) | second;
}

fn feature_vertex(first: u32, second: u32) -> u32 {
    return feature_pair(FEATURE_VERTEX, first, second);
}

fn feature_edge(first: u32, second: u32) -> u32 {
    return feature_pair(FEATURE_EDGE, first, second);
}

fn feature_face(first: u32, second: u32) -> u32 {
    return feature_pair(FEATURE_FACE, first, second);
}

fn feature_triangle(triangle: u32) -> u32 {
    return FEATURE_TRIANGLE | triangle;
}

fn feature_field_clip(id: u32) -> u32 {
    return FEATURE_CLIP | id;
}

fn feature_field_face(id: u32) -> u32 {
    return FEATURE_FACE_BIT | id;
}

fn feature_mirror(feature: u32) -> u32 {
    let kind = feature & FEATURE_KIND_MASK;
    if (kind == FEATURE_POINT) {
        return feature;
    }
    if (kind == FEATURE_TRIANGLE) {
        return feature ^ FEATURE_TRIANGLE_SIDE;
    }
    let first = (feature >> FEATURE_FIELD_BITS) & FEATURE_FIELD_MASK;
    let second = feature & FEATURE_FIELD_MASK;
    return kind | (second << FEATURE_FIELD_BITS) | first;
}

fn contact_mirror_features(contact: ptr<function, Contact>) {
    for (var index = 0u; index < (*contact).point_count; index = index + 1u) {
        (*contact).points[index].feature = feature_mirror((*contact).points[index].feature);
    }
}

fn manifold_candidate(position: vec3f, depth: f32, feature: u32) -> ManifoldPoint {
    return ManifoldPoint(position, depth, vec3f(0.0), vec3f(0.0), 0.0, 0.0, 0.0, feature);
}

fn manifold_push(contact: ptr<function, Contact>, point: vec3f, depth: f32, feature: u32) {
    let count = (*contact).point_count;
    if (count >= CONTACT_MAX_POINTS) {
        return;
    }
    (*contact).points[count] = manifold_candidate(point, depth, feature);
    (*contact).point_count = count + 1u;
}

fn manifold_anchor(contact: ptr<function, Contact>, first: Body, second: Body) {
    for (var index = 0u; index < (*contact).point_count; index = index + 1u) {
        let point = (*contact).points[index].position;
        (*contact).points[index].local_a = quat_rotate(quat_conjugate(first.state.orientation), point - first.state.position);
        (*contact).points[index].local_b = quat_rotate(quat_conjugate(second.state.orientation), point - second.state.position);
    }
}

fn manifold_arm(contact: Contact, body: Body, point: ManifoldPoint, first: bool) -> vec3f {
    if (first) {
        return quat_rotate(body.state.orientation, point.local_a) + body.state.position;
    }
    return quat_rotate(body.state.orientation, point.local_b) + body.state.position;
}

