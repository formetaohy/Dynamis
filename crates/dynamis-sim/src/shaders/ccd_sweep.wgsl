@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read> colliders: array<Collider>;
@group(0) @binding(3) var<storage, read> pair_keys_hi: array<u32>;
@group(0) @binding(4) var<storage, read> pair_keys_lo: array<u32>;
@group(0) @binding(5) var<storage, read> pair_count: atomic<u32>;

fn world_shape_hit(origin: vec3f, direction: vec3f, extent: f32, body: RigidBody, collider: Collider, expand: f32) -> ShapeHit {
    let center = body.position + quat_rotate(body.orientation, collider.local_offset);
    if (collider.shape == SHAPE_SPHERE) {
        return ray_sphere(origin, direction, extent, center, collider.radius + expand);
    }
    if (collider.shape == SHAPE_BOX) {
        let q = quat_mul(body.orientation, collider.local_rotation);
        return ray_box(origin, direction, extent, center, q, collider.half_extents + expand);
    }
    let axis = quat_rotate(body.orientation, vec3f(0.0, 1.0, 0.0));
    return ray_capsule(origin, direction, extent, center, axis, collider.half_height, collider.radius + expand);
}

fn sweep_retreat(
    moving: RigidBody, moving_collider: Collider,
    static_body: RigidBody, static_collider: Collider,
) -> f32 {
    let motion = moving.position - moving.prev_position;
    let sweep_length = length(motion);
    let bound = min_radius(moving_collider) + min_radius(static_collider);
    if (sweep_length <= bound * 0.5) {
        return 1.0;
    }
    let hit = world_shape_hit(
        moving.prev_position, motion / sweep_length, sweep_length,
        static_body, static_collider, min_radius(moving_collider),
    );
    if (hit.distance <= 0.0 || hit.distance >= sweep_length) {
        return 1.0;
    }
    return min(hit.distance / (sweep_length * 0.98), 1.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= atomicLoad(&pair_count)) {
        return;
    }
    if (index > 0u && pair_keys_hi[index] == pair_keys_hi[index - 1u] && pair_keys_lo[index] == pair_keys_lo[index - 1u]) {
        return;
    }
    let first_slot = pair_keys_hi[index];
    let second_slot = pair_keys_lo[index];
    var first = bodies[first_slot];
    var second = bodies[second_slot];
    if (first.inverse_mass == 0.0 && (first.flags & BODY_KINEMATIC) == 0u &&
        second.inverse_mass == 0.0 && (second.flags & BODY_KINEMATIC) == 0u) {
        return;
    }
    if ((first.collision_group & second.collision_mask) == 0u ||
        (second.collision_group & first.collision_mask) == 0u) {
        return;
    }
    let first_collider = colliders[first_slot];
    let second_collider = colliders[second_slot];
    let first_time = sweep_retreat(first, first_collider, second, second_collider);
    if (first_time < 1.0) {
        let first_motion = first.position - first.prev_position;
        first.position = first.prev_position + first_motion * first_time;
        bodies[first_slot] = first;
    }
    let second_time = sweep_retreat(second, second_collider, first, first_collider);
    if (second_time < 1.0) {
        let second_motion = second.position - second.prev_position;
        second.position = second.prev_position + second_motion * second_time;
        bodies[second_slot] = second;
    }
}
