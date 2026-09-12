struct Body {
    state: BodyState,
    desc: BodyDescriptor,
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

fn global_index(gid: vec3u) -> u32 {
    return gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
}

fn grid_stride(groups: vec3u) -> u32 {
    return groups.x * groups.y * WORKGROUP_SIZE;
}

fn cell_hash(coord: vec3i) -> u32 {
    let x = u32(coord.x) * 0x9E3779B9u;
    let y = u32(coord.y) * 0x85EBCA77u;
    let z = u32(coord.z) * 0xC2B2AE3Du;
    return (x ^ y ^ z ^ (x << 7u) ^ (y >> 3u) ^ (z << 11u)) & CELL_HASH_MASK;
}

fn cell_key(level: u32, coord: vec3i) -> u32 {
    return (level << LEVEL_KEY_SHIFT) | cell_hash(coord);
}

fn level_cell_size(level: u32, cell_size: f32) -> f32 {
    return cell_size * f32(1u << level);
}

fn cell_spans(bounds: Aabb, cell_size: f32) -> vec3i {
    return vec3i(floor(bounds.max / cell_size)) - vec3i(floor(bounds.min / cell_size)) + vec3i(1);
}

fn shape_levels(bounds: Aabb, cell_size: f32) -> u32 {
    var level = 0u;
    loop {
        let span = cell_spans(bounds, level_cell_size(level, cell_size));
        if (max(max(span.x, span.y), span.z) <= i32(MAX_CELLS_PER_AXIS) || level >= 30u) {
            break;
        }
        level = level + 1u;
    }
    return level;
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

fn manifold_push(contact: ptr<function, Contact>, point: vec3f, depth: f32) {
    let count = (*contact).point_count;
    if (count >= CONTACT_MAX_POINTS) {
        return;
    }
    (*contact).points[count] = ManifoldPoint(point, depth, 0.0, 0.0, 0.0, 0.0);
    (*contact).point_count = count + 1u;
}

