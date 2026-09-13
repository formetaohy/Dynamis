@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> colliders: array<Collider>;
@group(0) @binding(4) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(5) var<storage, read_write> aabbs: array<Aabb>;
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;

fn counter_max(slot: u32, value: u32) {
    atomicMax(&counters[slot * COUNTER_STRIDE_WORDS], value);
}

fn tight_bounds(state: BodyState, collider: Collider) -> Aabb {
    return world_aabb_of(world_collider(state, collider));
}

fn record_grid_resolution(tight: Aabb) {
    counter_max(COUNTER_GRID_SCALE, bitcast<u32>(grid_scale_of(tight)));
    counter_max(COUNTER_GRID_EXTENT, bitcast<u32>(max(grid_extent_of(tight), 0.0)));
}

fn swept_bounds(tight: Aabb, state: BodyState, desc: BodyDescriptor, collider: Collider) -> Aabb {
    if (!body_is_movable(desc)) {
        return tight;
    }
    let travel = state.velocity * params.dt;
    let margin = params.contact_margin + length(travel);
    let extent = (tight.max - tight.min) * 0.5 + vec3f(margin);
    let center = (tight.min + tight.max) * 0.5;
    let previous_center = state.prev_position + quat_rotate(state.orientation, collider.local_offset);
    var swept: Aabb;
    swept.min = min(center + travel, previous_center) - extent;
    swept.max = max(center + travel, previous_center) + extent;
    return swept;
}

fn work(index: u32) {
    let owner = collider_owners[index];
    if (owner == NO_BODY) {
        return;
    }
    let state = body_states[owner];
    let collider = colliders[index];
    let tight = tight_bounds(state, collider);
    record_grid_resolution(tight);
    aabbs[index] = swept_bounds(tight, state, body_descs[owner], collider);
}
