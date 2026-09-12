@group(0) @binding(3) var<storage, read> queries: array<Query>;
@group(0) @binding(4) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(5) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(6) var<storage, read> colliders: array<Collider>;
@group(0) @binding(7) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(8) var<storage, read_write> query_results: array<QueryResult>;
@group(0) @binding(9) var<uniform> params: StepParams;
@group(0) @binding(10) var<storage, read> collider_owners: array<u32>;

const CANDIDATES_PER_QUERY: u32 = 4096u;
const CELLS_PER_QUERY: u32 = 4096u;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

var<workgroup> candidates: array<u32, CANDIDATES_PER_QUERY>;
var<workgroup> candidate_count: atomic<u32>;
var<workgroup> overflow_flag: atomic<u32>;

fn bitonic_sort(local_invocation: u32) {
    var k = 2u;
    loop {
        if (k > CANDIDATES_PER_QUERY) {
            break;
        }
        var j = k / 2u;
        loop {
            if (j == 0u) {
                break;
            }
            var index = local_invocation;
            while (index < CANDIDATES_PER_QUERY) {
                let i = index ^ j;
                if (i > index) {
                    let ascending = (index & k) == 0u;
                    let a = candidates[index];
                    let b = candidates[i];
                    var out_of_order = a > b;
                    if (!ascending) {
                        out_of_order = a < b;
                    }
                    if (out_of_order) {
                        candidates[index] = b;
                        candidates[i] = a;
                    }
                }
                index = index + WORKGROUP_SIZE;
            }
            workgroupBarrier();
            j = j / 2u;
        }
        k = k * 2u;
    }
}

fn collect_cell(range: vec2u) {
    for (var entry = range.x; entry < range.y; entry = entry + 1u) {
        let slot = atomicAdd(&candidate_count, 1u);
        if (slot >= CANDIDATES_PER_QUERY) {
            atomicStore(&overflow_flag, 1u);
            continue;
        }
        candidates[slot] = entry_colliders[entry];
    }
}

fn collect_level_entries(level: u32, query_box: Aabb, invocation: u32) {
    let live = entry_live();
    let first = entry_bounds(live, level << LEVEL_KEY_SHIFT).x;
    let end = entry_bounds(live, (level + 1u) << LEVEL_KEY_SHIFT).x;
    var entry = first + invocation;
    while (entry < end && atomicLoad(&candidate_count) <= CANDIDATES_PER_QUERY) {
        let collider = entry_colliders[entry];
        if (aabb_overlaps(aabbs[collider], query_box)) {
            let slot = atomicAdd(&candidate_count, 1u);
            if (slot >= CANDIDATES_PER_QUERY) {
                atomicStore(&overflow_flag, 1u);
            } else {
                candidates[slot] = collider;
            }
        }
        entry = entry + WORKGROUP_SIZE;
    }
}

fn collect_level_cells(min_cell: vec3i, span: vec3i, level: u32, invocation: u32) {
    let rows = u32(span.x) * u32(span.y);
    let cells = rows * u32(span.z);
    var cell_index = invocation;
    while (cell_index < cells) {
        let dx = min_cell.x + i32(cell_index % u32(span.x));
        let dy = min_cell.y + i32((cell_index / u32(span.x)) % u32(span.y));
        let dz = min_cell.z + i32(cell_index / rows);
        collect_cell(entry_bounds_of(level, vec3i(dx, dy, dz)));
        cell_index = cell_index + WORKGROUP_SIZE;
    }
}

fn fits_cell_budget(span: vec3i) -> bool {
    if (span.x <= 0 || span.y <= 0 || span.z <= 0) {
        return false;
    }
    let axis_limit = i32(CELLS_PER_QUERY);
    if (span.x > axis_limit || span.y > axis_limit || span.z > axis_limit) {
        return false;
    }
    let rows = u32(span.x) * u32(span.y);
    return rows <= CELLS_PER_QUERY && rows * u32(span.z) <= CELLS_PER_QUERY;
}

fn collect_level(level: u32, query_box: Aabb, invocation: u32) {
    let cell_size = level_cell_size(level, grid_base_cell());
    let min_cell = vec3i(floor(query_box.min / cell_size));
    let max_cell = vec3i(floor(query_box.max / cell_size));
    let span = max_cell - min_cell + vec3i(1);
    if (fits_cell_budget(span)) {
        collect_level_cells(min_cell, span, level, invocation);
    } else {
        collect_level_entries(level, query_box, invocation);
    }
}

fn ray_hit(query: Query, body: Body, collider: Collider) -> ShapeHit {
    let direction = normalize(query.direction);
    let world = world_collider(body.state, collider);
    if (collider.kind == SHAPE_PLANE) {
        let n = plane_normal(world);
        let denom = dot(direction, n);
        if (abs(denom) < 1e-8) {
            return no_hit();
        }
        var t = dot(world.center - query.origin, n) / denom;
        if (t < 0.0 || t > query.extent) {
            return no_hit();
        }
        let point = query.origin + direction * t;
        return ShapeHit(t, point, n);
    }
    if (collider.kind == SHAPE_MESH || collider.kind == SHAPE_HEIGHTFIELD || collider.kind == SHAPE_HULL) {
        return ray_scene(world, query.origin, direction, query.extent);
    }
    return ray_scaled_shape(world, query.origin, direction, query.extent, 0.0);
}

fn query_shape_world(query: Query, center: vec3f) -> WorldShape {
    var world: WorldShape;
    world.kind = query.shape_kind;
    world.radius = query.radius;
    world.half_height = query.half_height;
    world.center = center;
    world.half_extents = query.half_extents;
    world.rotation = query.orientation;
    world.source = query.source;
    world.scale = vec3f(1.0);
    return world;
}

fn sweep_world_geom(query: Query, static_target: WorldShape, out_normal: ptr<function, vec3f>) -> f32 {
    let direction = normalize(query.direction);
    let start = query.origin;
    if (static_target.kind == SHAPE_PLANE) {
        let n = plane_normal(static_target);
        let signed = dot(start - static_target.center, n);
        let travel = dot(direction, n);
        if (travel >= 0.0) {
            return NO_HIT;
        }
        let time = max((signed - query.radius) / (-travel), 0.0);
        if (time > query.extent) {
            return NO_HIT;
        }
        *out_normal = n;
        return time;
    }
    let moving = query_shape_world(query, start);
    let hit = scene_sweep_hit(static_target, moving, start, direction, query.extent);
    if (hit.distance == NO_HIT) {
        return NO_HIT;
    }
    *out_normal = hit.normal;
    return hit.distance;
}

fn sweep_convex(query: Query, static_target: WorldShape, out_normal: ptr<function, vec3f>) -> f32 {
    let direction = normalize(query.direction);
    var t = 0.0;
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    var normal = sign_normalize(static_target.center - query.origin);
    for (var iter = 0u; iter < 8u; iter = iter + 1u) {
        let moved = query_shape_world(query, query.origin + direction * t);
        let closest = convex_closest(moved, static_target, &simplex, &count);
        if (closest.penetrating) {
            *out_normal = normal;
            return t;
        }
        if (closest.distance < 1e-4) {
            *out_normal = normal;
            return t;
        }
        normal = closest.normal;
        t = t + closest.distance;
        if (t > query.extent) {
            return NO_HIT;
        }
    }
    return NO_HIT;
}

fn overlap_hit(query: Query, body: Body, collider: Collider, out_normal: ptr<function, vec3f>) -> f32 {
    let world = world_collider(body.state, collider);
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    var probe = query_shape_world(query, query.origin);
    probe.radius = 0.0;
    let closest = convex_closest(probe, world, &simplex, &count);
    if (closest.penetrating) {
        let hit = convex_hit(query_shape_world(query, query.origin), world);
        *out_normal = hit.normal;
        return -query.extent + max(hit.distance + query.extent, 0.0);
    }
    if (closest.distance <= query.extent) {
        *out_normal = closest.normal;
        return closest.distance - query.extent;
    }
    return NO_HIT;
}

fn overlap_world_geom(query: Query, body: Body, collider: Collider, out_normal: ptr<function, vec3f>) -> f32 {
    let scene = world_collider(body.state, collider);
    if (collider.kind == SHAPE_PLANE) {
        let n = plane_normal(scene);
        let center_dist = dot(query.origin - scene.center, n);
        *out_normal = n;
        return center_dist - query.extent;
    }
    let probe = query_shape_world(query, query.origin);
    var triangle = 0u;
    let closest = scene_convex_closest(scene, probe, &triangle);
    *out_normal = closest.normal;
    if (closest.penetrating) {
        return -closest.distance;
    }
    let margin = scene_plane_margin(scene, probe, &triangle);
    return margin - query.extent;
}

fn body_passes(body: Body, collider: Collider, query: Query) -> bool {
    if (query.exclude_id != NO_BODY && body.state.body_id == query.exclude_id && body.state.generation == query.exclude_generation) {
        return false;
    }
    if (query.include_id != NO_BODY && (body.state.body_id != query.include_id || body.state.generation != query.include_generation)) {
        return false;
    }
    if ((query.filter_flags & FILTER_IGNORE_SENSORS) != 0u && (collider.flags & COLLIDER_SENSOR) != 0u) {
        return false;
    }
    if ((query.filter_flags & FILTER_IGNORE_SLEEPING) != 0u && body.state.sleeping != 0u) {
        return false;
    }
    if ((query.filter_flags & FILTER_IGNORE_STATIC) != 0u && body_is_static(body)) {
        return false;
    }
    if ((query.filter_flags & FILTER_IGNORE_KINEMATIC) != 0u && body_is_kinematic(body)) {
        return false;
    }
    if (!collider_filter_query(query, body, collider)) {
        return false;
    }
    if (collider.kind == SHAPE_NONE) {
        return false;
    }
    return true;
}

fn query_world_aabb(query: Query) -> Aabb {
    if (query.kind == QUERY_RAY) {
        let direction = normalize(query.direction);
        var aabb: Aabb;
        aabb.min = min(query.origin, query.origin + direction * query.extent);
        aabb.max = max(query.origin, query.origin + direction * query.extent);
        return aabb;
    }
    if (query.kind == QUERY_SWEEP) {
        var shape = query_shape_world(query, query.origin);
        var aabb = world_aabb_of(shape);
        let direction = normalize(query.direction);
        var end_aabb = world_aabb_of(query_shape_world(query, query.origin + direction * query.extent));
        aabb.min = min(aabb.min, end_aabb.min);
        aabb.max = max(aabb.max, end_aabb.max);
        return aabb;
    }
    let shape = query_shape_world(query, query.origin);
    return world_aabb_of(shape);
}

const NO_KEY: u32 = 0xFFFFFFFFu;
const NO_CANDIDATE: u32 = 0xFFFFFFFFu;
const CANDIDATES_PER_THREAD: u32 = (CANDIDATES_PER_QUERY + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

fn hit_key(distance: f32) -> u32 {
    let bits = bitcast<u32>(distance);
    if ((bits & 0x80000000u) != 0u) {
        return ~bits;
    }
    return bits | 0x80000000u;
}

fn candidate_index_of(invocation: u32, slot: u32) -> u32 {
    return invocation + slot * WORKGROUP_SIZE;
}

fn resolve_hit(query: Query, body: Body, collider: Collider) -> ShapeHit {
    if (query.kind == QUERY_RAY) {
        return ray_hit(query, body, collider);
    }
    var normal = vec3f(0.0);
    let is_world_geom = collider.kind == SHAPE_MESH || collider.kind == SHAPE_HEIGHTFIELD || collider.kind == SHAPE_PLANE;
    if (query.kind == QUERY_SWEEP) {
        let static_target = world_collider(body.state, collider);
        let distance = select(
            sweep_convex(query, static_target, &normal),
            sweep_world_geom(query, static_target, &normal),
            is_world_geom,
        );
        if (distance < NO_HIT) {
            return ShapeHit(distance, query.origin + normalize(query.direction) * distance, normal);
        }
        return no_hit();
    }
    if (query.kind == QUERY_SPHERE || query.kind == QUERY_POINT) {
        let separation = select(
            overlap_hit(query, body, collider, &normal),
            overlap_world_geom(query, body, collider, &normal),
            is_world_geom,
        );
        if (separation < NO_HIT) {
            return ShapeHit(separation, world_collider(body.state, collider).center, normal);
        }
        return no_hit();
    }
    if (is_world_geom) {
        let separation = overlap_world_geom(query, body, collider, &normal);
        if (separation < NO_HIT) {
            return ShapeHit(separation, world_collider(body.state, collider).center, normal);
        }
        return no_hit();
    }
    let world = world_collider(body.state, collider);
    let probe = query_shape_world(query, query.origin);
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    let closest = convex_closest(probe, world, &simplex, &count);
    if (closest.penetrating) {
        return ShapeHit(0.0, (closest.point_a + closest.point_b) * 0.5, closest.normal);
    }
    return no_hit();
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) invocation_id: vec3u,
) {
    let batch = workgroup_id.y * WORKGROUPS_PER_ROW + workgroup_id.x;
    let query = queries[batch];
    if (invocation_id.x == 0u) {
        atomicStore(&query_results[batch].header.count, 0u);
        atomicStore(&query_results[batch].header.overflow, 0u);
        atomicStore(&query_results[batch].header.round_key, NO_KEY);
        atomicStore(&query_results[batch].header.round_collider, NO_KEY);
    }
    workgroupBarrier();
    atomicStore(&candidate_count, 0u);
    atomicStore(&overflow_flag, 0u);
    workgroupBarrier();
    let query_box = query_world_aabb(query);
    var occupied = counter_load(COUNTER_GRID_LEVELS);
    while (occupied != 0u) {
        let level = countTrailingZeros(occupied);
        occupied = occupied & (occupied - 1u);
        collect_level(level, query_box, invocation_id.x);
    }
    workgroupBarrier();
    if (invocation_id.x == 0u && atomicLoad(&overflow_flag) != 0u) {
        atomicStore(&query_results[batch].header.overflow, 1u);
    }
    let candidate_total = min(atomicLoad(&candidate_count), CANDIDATES_PER_QUERY);
    for (var index = invocation_id.x; index < CANDIDATES_PER_QUERY; index = index + WORKGROUP_SIZE) {
        if (index >= candidate_total) {
            candidates[index] = NO_CANDIDATE;
        }
    }
    workgroupBarrier();
    bitonic_sort(invocation_id.x);
    workgroupBarrier();
    var keys: array<u32, CANDIDATES_PER_THREAD>;
    var best_key = NO_KEY;
    var best_collider = NO_KEY;
    var candidate_index = invocation_id.x;
    var slot_index = 0u;
    while (candidate_index < candidate_total) {
        let collider_slot = candidates[candidate_index];
        let duplicate = candidate_index > 0u && collider_slot == candidates[candidate_index - 1u];
        keys[slot_index] = NO_KEY;
        if (!duplicate) {
            let body = load_body(collider_owners[collider_slot]);
            let collider = colliders[collider_slot];
            if (body_passes(body, collider, query)) {
                let hit = resolve_hit(query, body, collider);
                if (hit.distance < NO_HIT) {
                    let key = hit_key(hit.distance);
                    keys[slot_index] = key;
                    atomicAdd(&query_results[batch].header.count, 1u);
                    if (key < best_key) {
                        best_key = key;
                        best_collider = collider_slot;
                    } else if (key == best_key) {
                        best_collider = min(best_collider, collider_slot);
                    }
                }
            }
        }
        slot_index = slot_index + 1u;
        candidate_index = candidate_index + WORKGROUP_SIZE;
    }
    let capacity = min(query.max_hits, MAX_HITS_PER_QUERY);
    storageBarrier();
    let total_hits = atomicLoad(&query_results[batch].header.count);
    let emitted_total = min(total_hits, capacity);
    var round = 0u;
    loop {
        if (round >= emitted_total) {
            break;
        }
        if (invocation_id.x == 0u) {
            atomicStore(&query_results[batch].header.round_key, NO_KEY);
            atomicStore(&query_results[batch].header.round_collider, NO_KEY);
        }
        storageBarrier();
        atomicMin(&query_results[batch].header.round_key, best_key);
        storageBarrier();
        let chosen_key = atomicLoad(&query_results[batch].header.round_key);
        if (chosen_key == NO_KEY) {
            break;
        }
        if (best_key == chosen_key) {
            atomicMin(&query_results[batch].header.round_collider, best_collider);
        }
        storageBarrier();
        let chosen_collider = atomicLoad(&query_results[batch].header.round_collider);
        if (best_key == chosen_key && best_collider == chosen_collider) {
            let body = load_body(collider_owners[chosen_collider]);
            let collider = colliders[chosen_collider];
            let hit = resolve_hit(query, body, collider);
            query_results[batch].hits[round] = QueryHit(body.state.body_id, body.state.generation, hit.distance, collider.slot, hit.point, 0.0, hit.normal, 0.0);
            best_key = NO_KEY;
            best_collider = NO_KEY;
            for (var slot = 0u; slot < slot_index; slot = slot + 1u) {
                if (keys[slot] == chosen_key) {
                    keys[slot] = NO_KEY;
                }
                if (keys[slot] < best_key) {
                    best_key = keys[slot];
                    best_collider = candidates[candidate_index_of(invocation_id.x, slot)];
                } else if (keys[slot] == best_key) {
                    best_collider = min(best_collider, candidates[candidate_index_of(invocation_id.x, slot)]);
                }
            }
        }
        round = round + 1u;
        storageBarrier();
    }
    if (invocation_id.x == 0u) {
        let spilling = atomicLoad(&query_results[batch].header.overflow) != 0u;
        atomicStore(&query_results[batch].header.count, emitted_total);
        atomicStore(&query_results[batch].header.overflow, select(0u, 1u, spilling || emitted_total < total_hits));
    }
}
