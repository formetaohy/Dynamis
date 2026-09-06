@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read> colliders: array<Collider>;
@group(0) @binding(3) var<storage, read_write> aabbs: array<Aabb>;

fn world_aabb(body: RigidBody, collider: Collider) -> Aabb {
    let offset = quat_rotate(body.orientation, collider.local_offset);
    let center = body.position + offset;
    var extent = vec3f(0.0);
    if (collider.shape == SHAPE_SPHERE) {
        extent = vec3f(collider.radius);
    } else if (collider.shape == SHAPE_BOX) {
        let q = quat_mul(body.orientation, collider.local_rotation);
        let e = collider.half_extents;
        extent = abs(quat_rotate(q, vec3f(e.x, 0.0, 0.0)))
            + abs(quat_rotate(q, vec3f(0.0, e.y, 0.0)))
            + abs(quat_rotate(q, vec3f(0.0, 0.0, e.z)));
    } else {
        let axis = quat_rotate(body.orientation, vec3f(0.0, 1.0, 0.0));
        extent = abs(axis * collider.half_height) + vec3f(collider.radius);
    }
    var aabb: Aabb;
    let prev_center = body.prev_position + quat_rotate(body.orientation, collider.local_offset);
    aabb.min = min(center - extent, prev_center - extent);
    aabb.max = max(center + extent, prev_center + extent);
    return aabb;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.body_count) {
        return;
    }
    let body = bodies[index];
    let collider = colliders[index];
    aabbs[index] = world_aabb(body, collider);
}
