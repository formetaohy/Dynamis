@group(0) @binding(0) var<storage, read> query_records: array<Query>;
@group(0) @binding(1) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read> colliders: array<Collider>;
@group(0) @binding(3) var<storage, read_write> query_results: array<QueryResult>;
@group(0) @binding(4) var<uniform> params: SimParams;

var<workgroup> partial_t: array<f32, WORKGROUP_SIZE>;
var<workgroup> partial_id: array<u32, WORKGROUP_SIZE>;
var<workgroup> partial_generation: array<u32, WORKGROUP_SIZE>;
var<workgroup> partial_point: array<vec3f, WORKGROUP_SIZE>;
var<workgroup> partial_normal: array<vec3f, WORKGROUP_SIZE>;

fn sphere_overlap_sphere(query: Query, body: RigidBody, collider: Collider) -> ShapeHit {
    let center = body.position + quat_rotate(body.orientation, collider.local_offset);
    let separation = length(query.origin - center) - query.extent - collider.radius;
    if (separation > 0.0) {
        return no_hit();
    }
    let normal = sign_normalize(center - query.origin);
    let point = center - normal * collider.radius;
    return ShapeHit(separation, point, normal);
}

fn sphere_overlap_box(query: Query, body: RigidBody, collider: Collider) -> ShapeHit {
    let closest = closest_point_box(query.origin, body, collider);
    let separation = length(query.origin - closest) - query.extent;
    if (separation > 0.0) {
        return no_hit();
    }
    let normal = sign_normalize(closest - query.origin);
    return ShapeHit(separation, closest, normal);
}

fn sphere_overlap_capsule(query: Query, body: RigidBody, collider: Collider) -> ShapeHit {
    let center = body.position + quat_rotate(body.orientation, collider.local_offset);
    let axis = quat_rotate(body.orientation, vec3f(0.0, 1.0, 0.0));
    let seg = Segment(center - axis * collider.half_height, center + axis * collider.half_height);
    let closest = closest_point_segment(query.origin, seg.start, seg.end);
    let separation = length(query.origin - closest) - query.extent - collider.radius;
    if (separation > 0.0) {
        return no_hit();
    }
    let normal = sign_normalize(closest - query.origin);
    return ShapeHit(separation, closest, normal);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) invocation_id: vec3u,
) {
    let query = query_records[workgroup_id.x];
    var best_t = NO_HIT;
    var best_id = NO_BODY;
    var best_generation = NO_BODY;
    var best_point = vec3f(0.0);
    var best_normal = vec3f(0.0);
    var index = invocation_id.x;
    while (index < params.body_count) {
        let body = bodies[index];
        let collider = colliders[index];
        var hit = no_hit();
        if (query.kind == QUERY_RAY) {
            let direction = normalize(query.direction);
            if (collider.shape == SHAPE_SPHERE) {
                let center = body.position + quat_rotate(body.orientation, collider.local_offset);
                hit = ray_sphere(query.origin, direction, query.extent, center, collider.radius);
            } else if (collider.shape == SHAPE_BOX) {
                let q = quat_mul(body.orientation, collider.local_rotation);
                hit = ray_box(query.origin, direction, query.extent, box_center(body, collider), q, collider.half_extents);
            } else {
                let center = body.position + quat_rotate(body.orientation, collider.local_offset);
                let axis = quat_rotate(body.orientation, vec3f(0.0, 1.0, 0.0));
                hit = ray_capsule(query.origin, direction, query.extent, center, axis, collider.half_height, collider.radius);
            }
        } else {
            if (collider.shape == SHAPE_SPHERE) {
                hit = sphere_overlap_sphere(query, body, collider);
            } else if (collider.shape == SHAPE_BOX) {
                hit = sphere_overlap_box(query, body, collider);
            } else {
                hit = sphere_overlap_capsule(query, body, collider);
            }
        }
        if (hit.distance < best_t) {
            best_t = hit.distance;
            best_id = body.body_id;
            best_generation = body.generation;
            best_point = hit.point;
            best_normal = hit.normal;
        }
        index = index + WORKGROUP_SIZE;
    }
    partial_t[invocation_id.x] = best_t;
    partial_id[invocation_id.x] = best_id;
    partial_generation[invocation_id.x] = best_generation;
    partial_point[invocation_id.x] = best_point;
    partial_normal[invocation_id.x] = best_normal;
    workgroupBarrier();
    if (invocation_id.x == 0u) {
        var result_t = NO_HIT;
        var result_id = NO_BODY;
        var result_generation = NO_BODY;
        var result_point = vec3f(0.0);
        var result_normal = vec3f(0.0);
        for (var i = 0u; i < WORKGROUP_SIZE; i = i + 1u) {
            if (partial_t[i] < result_t) {
                result_t = partial_t[i];
                result_id = partial_id[i];
                result_generation = partial_generation[i];
                result_point = partial_point[i];
                result_normal = partial_normal[i];
            }
        }
        if (result_t == NO_HIT) {
            query_results[query.slot] = QueryResult(NO_BODY, NO_BODY, 0.0, 0u, vec3f(0.0), 0.0, vec3f(0.0), 0.0);
        } else {
            query_results[query.slot] = QueryResult(result_id, result_generation, result_t, 1u, result_point, 0.0, result_normal, 0.0);
        }
    }
}
