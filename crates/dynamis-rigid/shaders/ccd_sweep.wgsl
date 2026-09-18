@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> colliders: array<Collider>;
@group(0) @binding(4) var<storage, read> pair_major: array<u32>;
@group(0) @binding(5) var<storage, read> pair_minor: array<u32>;
@group(0) @binding(6) var<storage, read_write> pair_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(8) var<storage, read_write> ccd_factor: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> ccd_impact: array<CcdImpact>;
@group(0) @binding(10) var<storage, read> joint_major: array<u32>;
@group(0) @binding(11) var<storage, read> joint_minor: array<u32>;
@group(0) @binding(12) var<storage, read_write> joint_count: array<atomic<u32>>;

const SWEEP_ITERATIONS: u32 = 16u;

struct SweepReach {
    travel: f32,
    spin: f32,
}

struct SweepHit {
    time: f32,
    normal: vec3f,
    point: vec3f,
}

fn no_sweep_hit() -> SweepHit {
    var hit: SweepHit;
    hit.time = 1.0;
    hit.normal = vec3f(0.0, 1.0, 0.0);
    hit.point = vec3f(0.0);
    return hit;
}

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn sweep_pose(state: BodyState, t: f32) -> BodyState {
    var pose = state;
    pose.position = state.prev_position + (state.position - state.prev_position) * t;
    pose.orientation = quat_slerp(state.prev_orientation, state.orientation, t);
    return pose;
}

fn sweep_radius(state: BodyState, desc: BodyDescriptor, collider: Collider) -> f32 {
    if (collider.kind == SHAPE_PLANE) {
        return 0.0;
    }
    let bounds = world_aabb_of(world_collider(state, collider));
    let center = state.position + quat_rotate(state.orientation, collider.local_offset);
    return length((bounds.max - bounds.min) * 0.5) + length(center - body_com_of(state, desc));
}

fn sweep_reach(state: BodyState, desc: BodyDescriptor, collider: Collider) -> SweepReach {
    var reach: SweepReach;
    reach.travel = length(state.position - state.prev_position);
    reach.spin = quat_angle(state.prev_orientation, state.orientation) * sweep_radius(state, desc, collider);
    return reach;
}

fn plane_convex_gap(plane: WorldShape, convex: WorldShape) -> ShapeHit {
    let normal = plane_normal(plane);
    let side = dot(convex.center - plane.center, normal);
    let facing = select(normal, -normal, side < 0.0);
    let extent = dot(support(convex, facing) - convex.center, facing);
    return ShapeHit(abs(side) - extent, convex.center + facing * extent, facing, NO_TRIANGLE);
}

fn pair_gap(first: WorldShape, second: WorldShape) -> ShapeHit {
    if (first.kind == SHAPE_PLANE) {
        let swapped = plane_convex_gap(first, second);
        return ShapeHit(swapped.distance, swapped.point, -swapped.normal, NO_TRIANGLE);
    }
    if (second.kind == SHAPE_PLANE) {
        return plane_convex_gap(second, first);
    }
    let first_scene = shape_triangle_scene(first.kind);
    let second_scene = shape_triangle_scene(second.kind);
    if (first_scene && second_scene) {
        return no_hit();
    }
    if (second_scene) {
        let closest = scene_gap(second, first);
        if (closest.distance == NO_HIT) {
            return no_hit();
        }
        return ShapeHit(closest.distance, closest.point, -closest.normal, closest.triangle);
    }
    if (first_scene) {
        let closest = scene_gap(first, second);
        if (closest.distance == NO_HIT) {
            return no_hit();
        }
        return ShapeHit(closest.distance, closest.point, closest.normal, closest.triangle);
    }
    return convex_hit(first, second);
}

fn rigid_sweep(moving: Body, moving_collider: Collider, other: Body, other_collider: Collider) -> SweepHit {
    if (!body_has_ccd(moving)) {
        return no_sweep_hit();
    }
    let free_reach = sweep_reach(moving.state, moving.desc, moving_collider);
    let other_movable = body_is_movable(other.desc);
    var other_reach: SweepReach;
    other_reach.travel = 0.0;
    other_reach.spin = 0.0;
    if (other_movable) {
        other_reach = sweep_reach(other.state, other.desc, other_collider);
    }
    let closing_floor = free_reach.spin + other_reach.spin;
    if (free_reach.travel + closing_floor + other_reach.travel <= 0.0) {
        return no_sweep_hit();
    }
    let free_travel = moving.state.position - moving.state.prev_position;
    var other_travel = vec3f(0.0);
    if (other_movable) {
        other_travel = other.state.position - other.state.prev_position;
    }
    var t = 0.0;
    for (var iteration = 0u; iteration < SWEEP_ITERATIONS; iteration = iteration + 1u) {
        let free = world_collider(sweep_pose(moving.state, t), moving_collider);
        var fixed = world_collider(other.state, other_collider);
        if (other_movable) {
            fixed = world_collider(sweep_pose(other.state, t), other_collider);
        }
        let gap = pair_gap(free, fixed);
        if (gap.distance <= SWEEP_TOLERANCE) {
            if (iteration == 0u && gap.distance <= 0.0) {
                return no_sweep_hit();
            }
            var hit: SweepHit;
            hit.time = t;
            hit.normal = gap.normal;
            hit.point = select(support(free, gap.normal), free.center, free.kind == SHAPE_PLANE);
            return hit;
        }
        let closing = dot(free_travel - other_travel, gap.normal) + closing_floor;
        if (closing <= 0.0) {
            return no_sweep_hit();
        }
        t = t + gap.distance / closing;
        if (t >= 1.0) {
            return no_sweep_hit();
        }
    }
    return no_sweep_hit();
}

fn publish_impact(slot: u32, time: f32, normal: vec3f, restitution: f32, point: vec3f) {
    let packed = bitcast<u32>(time);
    if (packed < atomicMin(&ccd_factor[slot], packed)) {
        var impact: CcdImpact;
        impact.normal = normal;
        impact.restitution = restitution;
        impact.point = point;
        impact._pad0 = 0.0;
        ccd_impact[slot] = impact;
    }
}

fn retreat(slot: u32, moving: Body, moving_collider: Collider, other: Body, other_collider: Collider) {
    let hit = rigid_sweep(moving, moving_collider, other, other_collider);
    if (hit.time >= 1.0) {
        return;
    }
    publish_impact(
        slot,
        hit.time,
        sign_normalize(hit.normal),
        material_combine(
            moving_collider.restitution,
            other_collider.restitution,
            params.restitution_combine,
        ),
        hit.point,
    );
}

fn work(index: u32) {
    if (index > 0u && pair_major[index] == pair_major[index - 1u] && pair_minor[index] == pair_minor[index - 1u]) {
        return;
    }
    let first_slot = pair_major[index];
    let second_slot = pair_minor[index];
    let first_body_slot = collider_owners[first_slot];
    let second_body_slot = collider_owners[second_slot];
    if (first_body_slot == NO_BODY || second_body_slot == NO_BODY || first_body_slot == second_body_slot) {
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
    if (joined_by_joint(first_body_slot, second_body_slot)) {
        return;
    }
    retreat(first_body_slot, first, first_collider, second, second_collider);
    retreat(second_body_slot, second, second_collider, first, first_collider);
}
