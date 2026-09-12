@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> colliders: array<Collider>;
@group(0) @binding(4) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(5) var<storage, read_write> aabbs: array<Aabb>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.collider_count) {
        return;
    }
    let collider = colliders[index];
    let owner = collider_owners[index];
    if (collider.kind == SHAPE_NONE || owner == NO_BODY || !body_is_movable(body_descs[owner])) {
        return;
    }
    let state = body_states[owner];
    let travel = state.velocity * params.dt;
    let margin = params.contact_margin + length(travel);
    let world = world_collider(state, collider);
    let aabb = world_aabb_of(world);
    let extent = (aabb.max - aabb.min) * 0.5 + vec3f(margin);
    let center = state.position + quat_rotate(state.orientation, collider.local_offset);
    let prev_center = state.prev_position + quat_rotate(state.orientation, collider.local_offset);
    var swept: Aabb;
    swept.min = min(center + travel, prev_center) - extent;
    swept.max = max(center + travel, prev_center) + extent;
    aabbs[index] = swept;
}
