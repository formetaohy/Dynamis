@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> colliders: array<Collider>;
@group(0) @binding(4) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(5) var<storage, read_write> aabbs: array<Aabb>;
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;

const HALF_TURN: f32 = 3.141592653589793;

fn tight_bounds(state: BodyState, collider: Collider) -> Aabb {
    return world_aabb_of(world_collider(state, collider));
}

fn resolution_bounds(state: BodyState, collider: Collider) -> Aabb {
    var upright = world_collider(state, collider);
    upright.rotation = vec4f(0.0, 0.0, 0.0, 1.0);
    return world_aabb_of(upright);
}

fn record_grid_resolution(bounds: Aabb) {
    counter_max(COUNTER_GRID_SCALE, bitcast<u32>(grid_scale_of(bounds)));
    counter_max(COUNTER_GRID_EXTENT, bitcast<u32>(max(grid_extent_of(bounds), 0.0)));
}

fn sweep_radius(state: BodyState, desc: BodyDescriptor, collider: Collider, tight: Aabb) -> f32 {
    let com = body_com_of(state, desc);
    let center = state.position + quat_rotate(state.orientation, collider.local_offset);
    return length((tight.max - tight.min) * 0.5) + length(center - com);
}

fn rotation_reach(spin: f32, radius: f32) -> f32 {
    if (spin >= HALF_TURN) {
        return 2.0 * radius;
    }
    return 2.0 * radius * sin(0.5 * spin);
}

fn swept_bounds(tight: Aabb, state: BodyState, desc: BodyDescriptor, collider: Collider) -> Aabb {
    if (!body_is_movable(desc)) {
        return tight;
    }
    let travel = abs(state.velocity * params.dt);
    let spin = length(state.angular_velocity * params.dt);
    let reach = vec3f(params.contact_margin + rotation_reach(spin, sweep_radius(state, desc, collider, tight)));
    var swept: Aabb;
    swept.min = tight.min - travel - reach;
    swept.max = tight.max + travel + reach;
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
    record_grid_resolution(resolution_bounds(state, collider));
    aabbs[index] = swept_bounds(tight, state, body_descs[owner], collider);
}
