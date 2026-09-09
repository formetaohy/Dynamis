@group(0) @binding(0) var<storage, read> queries: array<Query>;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> colliders: array<Collider>;
@group(0) @binding(4) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(5) var<storage, read> entry_keys_hi: array<u32>;
@group(0) @binding(6) var<storage, read> entry_keys_lo: array<u32>;
@group(0) @binding(7) var<storage, read_write> entry_count: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> query_results: array<QueryResult>;
@group(0) @binding(9) var<storage, read> large_bodies: array<u32>;
@group(0) @binding(10) var<storage, read_write> large_count: array<atomic<u32>>;
@group(0) @binding(11) var<uniform> params: SimParams;

const CANDIDATES_PER_QUERY: u32 = 512u;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

var<workgroup> candidates: array<u32, CANDIDATES_PER_QUERY>;
var<workgroup> candidate_count: atomic<u32>;
var<workgroup> overflow_flag: atomic<u32>;

fn bitonic_sort(local_invocation: u32) {
    let total = min(atomicLoad(&candidate_count), CANDIDATES_PER_QUERY);
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
                    let out_of_order = select(a > b, a < b, ascending);
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

fn cell_hash(coord: vec3i) -> u32 {
    let x = u32(coord.x) * 0x9E3779B9u;
    let y = u32(coord.y) * 0x85EBCA77u;
    let z = u32(coord.z) * 0xC2B2AE3Du;
    return x ^ y ^ z ^ (x << 7u) ^ (y >> 3u) ^ (z << 11u);
}

fn hash_range(hash: u32) -> vec2u {
    var lo = 0u;
    var hi = min(atomicLoad(&entry_count[0]), arrayLength(&entry_keys_lo));
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (entry_keys_hi[mid] < hash) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let first = lo;
    hi = min(atomicLoad(&entry_count[0]), arrayLength(&entry_keys_lo));
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (entry_keys_hi[mid] <= hash) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return vec2u(first, lo);
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
    let hit = scene_sweep_hit(moving, start, direction, static_target.source, static_target.scale, query.extent);
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
    let world = world_collider(body.state, collider);
    if (collider.kind == SHAPE_PLANE) {
        let n = plane_normal(world);
        let center_dist = dot(query.origin - world.center, n);
        *out_normal = n;
        return center_dist - query.extent;
    }
    let probe = query_shape_world(query, query.origin);
    var triangle = 0u;
    let closest = scene_convex_closest(collider.source, collider.scale, probe, &triangle);
    *out_normal = closest.normal;
    if (closest.penetrating) {
        return -closest.distance;
    }
    let margin = scene_plane_margin(collider.source, collider.scale, probe, &triangle);
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

fn emit_hit(batch: u32, query: Query, body: Body, collider: Collider, collider_index: u32, distance: f32, point: vec3f, normal: vec3f) {
    var slot = 0u;
    loop {
        let current = atomicLoad(&query_results[batch].header.count);
        if (current >= query.max_hits || current >= MAX_HITS_PER_QUERY) {
            atomicStore(&query_results[batch].header.overflow, 1u);
            return;
        }
        let exchanged = atomicCompareExchangeWeak(&query_results[batch].header.count, current, current + 1u);
        if (exchanged.exchanged) {
            slot = current;
            break;
        }
    }
    query_results[batch].hits[slot] = QueryHit(body.state.body_id, body.state.generation, distance, collider_index, point, 0.0, normal, 0.0);
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
    }
    workgroupBarrier();
    atomicStore(&candidate_count, 0u);
    atomicStore(&overflow_flag, 0u);
    workgroupBarrier();
    let query_box = query_world_aabb(query);
    let cell_size = params.grid_cell_size;
    let min_cell = vec3i(floor(query_box.min / cell_size));
    let max_cell = vec3i(floor(query_box.max / cell_size));
    let span = max_cell - min_cell + vec3i(1);
    let cells_total = span.x * span.y * span.z;
    var cell_index = invocation_id.x;
    while (cell_index < u32(cells_total)) {
        let dx = min_cell.x + i32(cell_index % u32(span.x));
        let dy = min_cell.y + i32((cell_index / u32(span.x)) % u32(span.y));
        let dz = min_cell.z + i32(cell_index / (u32(span.x) * u32(span.y)));
        let hash = cell_hash(vec3i(dx, dy, dz));
        let range = hash_range(hash);
        for (var entry = range.x; entry < range.y; entry = entry + 1u) {
            let slot = atomicAdd(&candidate_count, 1u);
            if (slot >= CANDIDATES_PER_QUERY) {
                atomicStore(&overflow_flag, 1u);
                continue;
            }
            candidates[slot] = entry_keys_lo[entry];
        }
        cell_index = cell_index + WORKGROUP_SIZE;
    }
    var large_index = invocation_id.x;
    while (large_index < min(atomicLoad(&large_count[0]), arrayLength(&large_bodies))) {
        let slot = atomicAdd(&candidate_count, 1u);
        if (slot < CANDIDATES_PER_QUERY) {
            candidates[slot] = large_bodies[large_index];
        } else {
            atomicStore(&overflow_flag, 1u);
        }
        large_index = large_index + WORKGROUP_SIZE;
    }
    workgroupBarrier();
    bitonic_sort(invocation_id.x);
    workgroupBarrier();
    let candidate_total = min(atomicLoad(&candidate_count), CANDIDATES_PER_QUERY);
    var candidate_index = invocation_id.x;
    while (candidate_index < candidate_total) {
        let collider_slot = candidates[candidate_index];
        if (candidate_index > 0u && collider_slot == candidates[candidate_index - 1u]) {
            candidate_index = candidate_index + WORKGROUP_SIZE;
            continue;
        }
        let body = load_body(collider_slot / MAX_COLLIDERS_PER_BODY);
        let collider = colliders[collider_slot];
        if (!body_passes(body, collider, query)) {
            candidate_index = candidate_index + WORKGROUP_SIZE;
            continue;
        }
        var hit = no_hit();
        var normal = vec3f(0.0);
        if (query.kind == QUERY_RAY) {
            hit = ray_hit(query, body, collider);
        } else if (query.kind == QUERY_SWEEP) {
            let static_target = world_collider(body.state, collider);
            let moving_probe = query_shape_world(query, query.origin);
            let is_world_geom = collider.kind == SHAPE_MESH || collider.kind == SHAPE_HEIGHTFIELD || collider.kind == SHAPE_PLANE;
            let distance = select(
                sweep_convex(query, static_target, &normal),
                sweep_world_geom(query, static_target, &normal),
                is_world_geom,
            );
            if (distance < NO_HIT) {
                let point = query.origin + normalize(query.direction) * distance;
                hit = ShapeHit(distance, point, normal);
            }
        } else if (query.kind == QUERY_SPHERE || query.kind == QUERY_POINT) {
            let is_world_geom = collider.kind == SHAPE_MESH || collider.kind == SHAPE_HEIGHTFIELD || collider.kind == SHAPE_PLANE;
            let separation = select(
                overlap_hit(query, body, collider, &normal),
                overlap_world_geom(query, body, collider, &normal),
                is_world_geom,
            );
            if (separation < NO_HIT) {
                let point = world_collider(body.state, collider).center;
                hit = ShapeHit(separation, point, normal);
            }
        } else {
            let is_world_geom = collider.kind == SHAPE_MESH || collider.kind == SHAPE_HEIGHTFIELD || collider.kind == SHAPE_PLANE;
            if (is_world_geom) {
                let separation = overlap_world_geom(query, body, collider, &normal);
                if (separation < NO_HIT) {
                    let point = world_collider(body.state, collider).center;
                    hit = ShapeHit(separation, point, normal);
                }
            } else {
                let world = world_collider(body.state, collider);
                let probe = query_shape_world(query, query.origin);
                var simplex: array<SimplexPoint, 4>;
                var count = 0u;
                let closest = convex_closest(probe, world, &simplex, &count);
                if (closest.penetrating) {
                    hit = ShapeHit(0.0, (closest.point_a + closest.point_b) * 0.5, closest.normal);
                }
            }
        }
        if (hit.distance < NO_HIT) {
            emit_hit(batch, query, body, collider, collider_slot % MAX_COLLIDERS_PER_BODY, hit.distance, hit.point, hit.normal);
        }
        candidate_index = candidate_index + WORKGROUP_SIZE;
    }
}
