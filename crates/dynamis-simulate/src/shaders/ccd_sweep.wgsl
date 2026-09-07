@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read> colliders: array<Collider>;
@group(0) @binding(3) var<storage, read> pair_keys_hi: array<u32>;
@group(0) @binding(4) var<storage, read> pair_keys_lo: array<u32>;
@group(0) @binding(5) var<storage, read_write> pair_count: atomic<u32>;

fn sweep_retreat(
    moving: RigidBody, moving_collider: Collider,
    static_body: RigidBody, static_collider: Collider,
) -> f32 {
    if (!body_has_ccd(moving)) {
        return 1.0;
    }
    let motion = moving.position - moving.prev_position;
    let sweep_length = length(motion);
    let bound = min_radius(moving_collider) + min_radius(static_collider);
    if (sweep_length <= bound * 0.5) {
        return 1.0;
    }
    let moving_world = world_collider(moving, moving_collider);
    let static_world = world_collider(static_body, static_collider);
    if (static_world.kind == SHAPE_NONE) {
        return 1.0;
    }
    let hit = convex_hit_at(
        moving_world,
        moving.prev_position,
        motion / sweep_length,
        static_world,
        min_radius(moving_collider),
        sweep_length,
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
    if (first_slot / 4u == second_slot / 4u) {
        return;
    }
    var first = bodies[first_slot / 4u];
    var second = bodies[second_slot / 4u];
    if (first.inverse_mass == 0.0 && (first.flags & BODY_KINEMATIC) == 0u &&
        second.inverse_mass == 0.0 && (second.flags & BODY_KINEMATIC) == 0u) {
        return;
    }
    if (!body_world_intersects(first, second.collision_group, second.collision_mask)) {
        return;
    }
    let first_collider = colliders[first_slot];
    let second_collider = colliders[second_slot];
    if (first_collider.kind == SHAPE_NONE || second_collider.kind == SHAPE_NONE) {
        return;
    }
    let first_time = sweep_retreat(first, first_collider, second, second_collider);
    if (first_time < 1.0) {
        let first_motion = first.position - first.prev_position;
        first.position = first.prev_position + first_motion * first_time;
        let first_world = world_collider(first, first_collider);
        let second_world = world_collider(second, second_collider);
        let axis = sign_normalize(second_world.center - first_world.center);
        let normal_speed = dot(first.velocity, axis);
        if (normal_speed > 0.0) {
            let restitution = material_combine(first_collider.restitution, second_collider.restitution, params.restitution_combine);
            first.velocity = first.velocity - axis * normal_speed * (1.0 + restitution);
        }
        bodies[first_slot / 4u] = first;
    }
    let second_time = sweep_retreat(second, second_collider, first, first_collider);
    if (second_time < 1.0) {
        let second_motion = second.position - second.prev_position;
        second.position = second.prev_position + second_motion * second_time;
        let first_world = world_collider(first, first_collider);
        let second_world = world_collider(second, second_collider);
        let axis = sign_normalize(first_world.center - second_world.center);
        let normal_speed = dot(second.velocity, axis);
        if (normal_speed > 0.0) {
            let restitution = material_combine(first_collider.restitution, second_collider.restitution, params.restitution_combine);
            second.velocity = second.velocity - axis * normal_speed * (1.0 + restitution);
        }
        bodies[second_slot / 4u] = second;
    }
}
