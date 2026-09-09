@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> colliders: array<Collider>;
@group(0) @binding(4) var<storage, read> pair_keys_hi: array<u32>;
@group(0) @binding(5) var<storage, read> pair_keys_lo: array<u32>;
@group(0) @binding(6) var<storage, read_write> pair_count: array<atomic<u32>>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn sweep_retreat(
    moving: Body, moving_collider: Collider,
    static_body: Body, static_collider: Collider,
) -> f32 {
    if (!body_has_ccd(moving)) {
        return 1.0;
    }
    let motion = moving.state.position - moving.state.prev_position;
    let sweep_length = length(motion);
    let bound = min_radius(moving_collider) + min_radius(static_collider);
    if (sweep_length <= bound * 0.5) {
        return 1.0;
    }
    let moving_world = world_collider(moving.state, moving_collider);
    let static_world = world_collider(static_body.state, static_collider);
    if (static_world.kind == SHAPE_NONE) {
        return 1.0;
    }
    let hit = convex_hit_at(
        moving_world,
        moving.state.prev_position,
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

fn retreat(slot: u32, moving: Body, moving_collider: Collider, other: Body, other_collider: Collider) {
    let time = sweep_retreat(moving, moving_collider, other, other_collider);
    if (time >= 1.0) {
        return;
    }
    var state = moving.state;
    state.position = state.prev_position + (state.position - state.prev_position) * time;
    let axis = sign_normalize(
        world_collider(other.state, other_collider).center - world_collider(state, moving_collider).center);
    let normal_speed = dot(state.velocity, axis);
    if (normal_speed > 0.0) {
        let restitution = material_combine(
            moving_collider.restitution,
            other_collider.restitution,
            params.restitution_combine,
        );
        state.velocity = state.velocity - axis * normal_speed * (1.0 + restitution);
    }
    body_states[slot] = state;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= min(atomicLoad(&pair_count[0]), arrayLength(&pair_keys_hi))) {
        return;
    }
    if (index > 0u && pair_keys_hi[index] == pair_keys_hi[index - 1u] && pair_keys_lo[index] == pair_keys_lo[index - 1u]) {
        return;
    }
    let first_slot = pair_keys_hi[index];
    let second_slot = pair_keys_lo[index];
    let first_body_slot = first_slot / MAX_COLLIDERS_PER_BODY;
    let second_body_slot = second_slot / MAX_COLLIDERS_PER_BODY;
    if (first_body_slot == second_body_slot) {
        return;
    }
    let first = load_body(first_body_slot);
    let second = load_body(second_body_slot);
    if (body_is_static(first) && body_is_static(second)) {
        return;
    }
    let first_collider = colliders[first_slot];
    let second_collider = colliders[second_slot];
    if (first_collider.kind == SHAPE_NONE || second_collider.kind == SHAPE_NONE) {
        return;
    }
    if (!collider_filter_intersects(first, first_collider, second, second_collider)) {
        return;
    }
    retreat(first_body_slot, first, first_collider, second, second_collider);
    retreat(second_body_slot, second, second_collider, first, first_collider);
}
