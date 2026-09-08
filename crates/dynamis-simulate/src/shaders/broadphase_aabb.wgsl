@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read> colliders: array<Collider>;
@group(0) @binding(3) var<storage, read_write> aabbs: array<Aabb>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.dynamic_count) {
        return;
    }
    let body = bodies[index];
    for (var i = 0u; i < MAX_COLLIDERS_PER_BODY; i = i + 1u) {
        let collider_index = index * MAX_COLLIDERS_PER_BODY + i;
        let collider = colliders[collider_index];
        if (collider.kind == SHAPE_NONE) {
            var empty: Aabb;
            empty.min = vec3f(3.402823466e38);
            empty.max = vec3f(-3.402823466e38);
            aabbs[collider_index] = empty;
            continue;
        }
        let world = world_collider(body, collider);
        var aabb = world_aabb_of(world);
        let extent = (aabb.max - aabb.min) * 0.5;
        let prev_center = body.prev_position + quat_rotate(body.orientation, collider.local_offset);
        aabb.min = min(aabb.min, prev_center - extent);
        aabb.max = max(aabb.max, prev_center + extent);
        aabbs[collider_index] = aabb;
    }
}
