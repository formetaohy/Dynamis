@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> colliders: array<Collider>;
@group(0) @binding(4) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(5) var<storage, read_write> aabbs: array<Aabb>;

fn swept_bounds(state: BodyState, desc: BodyDescriptor, collider: Collider) -> Aabb {
    let world = world_collider(state, collider);
    let bounds = world_aabb_of(world);
    if (!body_is_movable(desc)) {
        return bounds;
    }
    let travel = state.velocity * params.dt;
    let margin = params.contact_margin + length(travel);
    let extent = (bounds.max - bounds.min) * 0.5 + vec3f(margin);
    let previous_center = state.prev_position + quat_rotate(state.orientation, collider.local_offset);
    var swept: Aabb;
    swept.min = min(world.center + travel, previous_center) - extent;
    swept.max = max(world.center + travel, previous_center) + extent;
    return swept;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.collider_count) {
        return;
    }
    let owner = collider_owners[index];
    if (owner == NO_BODY) {
        return;
    }
    aabbs[index] = swept_bounds(body_states[owner], body_descs[owner], colliders[index]);
}
