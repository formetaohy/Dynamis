@group(0) @binding(0) var<storage, read> queries: array<Query>;
@group(0) @binding(1) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read> colliders: array<Collider>;
@group(0) @binding(3) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(4) var<storage, read> entry_keys_hi: array<u32>;
@group(0) @binding(5) var<storage, read> entry_keys_lo: array<u32>;
@group(0) @binding(6) var<storage, read> entry_count: array<u32>;
@group(0) @binding(7) var<storage, read_write> query_headers: array<QueryResultHeader>;
@group(0) @binding(8) var<storage, read_write> query_hits: array<QueryHit>;
@group(0) @binding(9) var<storage, read> large_bodies: array<u32>;
@group(0) @binding(10) var<storage, read> large_count: array<u32>;
@group(0) @binding(11) var<uniform> params: SimParams;

const CANDIDATES_PER_QUERY: u32 = 512u;

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
    var hi = entry_count[0];
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (entry_keys_hi[mid] < hash) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    var first = lo;
    hi = entry_count[0];
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

fn ray_hit(query: Query, body: RigidBody, collider: Collider) -> ShapeHit {
    let direction = normalize(query.direction);
    if (collider.kind == SHAPE_SPHERE) {
        let world = world_collider(body, collider);
        return ray_sphere(query.origin, direction, query.extent, world.center, world.radius);
    }
    if (collider.kind == SHAPE_CUBOID) {
        let world = world_collider(body, collider);
        return ray_box(query.origin, direction, query.extent, world.center, world.rotation, world.half_extents);
    }
    if (collider.kind == SHAPE_CAPSULE) {
        let world = world_collider(body, collider);
        return ray_capsule(query.origin, direction, query.extent, world.center, shape_axis(world), world.half_height, world.radius);
    }
    if (collider.kind == SHAPE_CYLINDER) {
        let world = world_collider(body, collider);
        return ray_cylinder(query.origin, direction, query.extent, world.center, shape_axis(world), world.half_height, world.radius);
    }
    let world = world_collider(body, collider);
    return ray_scene(world, query.origin, direction, query.extent);
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
    return world;
}

fn sweep_convex(query: Query, static_target: WorldShape, out_normal: ptr<function, vec3f>) -> f32 {
    let direction = normalize(query.direction);
    var t = 0.0;
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    for (var iter = 0u; iter < 8u; iter = iter + 1u) {
        let moved = query_shape_world(query, query.origin + direction * t);
        let closest = convex_closest(moved, static_target, &simplex, &count);
        if (closest.penetrating) {
            *out_normal = closest.normal;
            return t;
        }
        if (closest.distance < 1e-4) {
            *out_normal = closest.normal;
            return t;
        }
        t = t + closest.distance;
        if (t > query.extent) {
            return NO_HIT;
        }
    }
    return NO_HIT;
}

fn overlap_hit(query: Query, body: RigidBody, collider: Collider, out_normal: ptr<function, vec3f>) -> f32 {
    let world = world_collider(body, collider);
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

fn body_passes(body: RigidBody, collider: Collider, query: Query) -> bool {
    if ((query.filter_flags & FILTER_IGNORE_SENSORS) != 0u && (collider.flags & COLLIDER_SENSOR) != 0u) {
        return false;
    }
    if ((query.filter_flags & FILTER_IGNORE_SLEEPING) != 0u && (body.flags & BODY_SLEEPING) != 0u) {
        return false;
    }
    if ((query.filter_flags & FILTER_IGNORE_STATIC) != 0u && body_is_static(body)) {
        return false;
    }
    if ((query.filter_flags & FILTER_IGNORE_KINEMATIC) != 0u && (body.flags & BODY_KINEMATIC) != 0u) {
        return false;
    }
    if (query.group != 0u && !body_world_intersects(body, query.group, query.mask)) {
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

fn emit_hit(query: Query, body: RigidBody, collider: Collider, distance: f32, point: vec3f, normal: vec3f) {
    var slot = 0u;
    loop {
        let current = atomicLoad(&query_headers[query.slot].count);
        if (current >= query.max_hits || current >= MAX_HITS_PER_QUERY) {
            atomicStore(&query_headers[query.slot].overflow, 1u);
            return;
        }
        let result = atomicCompareExchangeWeak(&query_headers[query.slot].count, current, current + 1u);
        if (result.exchanged) {
            slot = current;
            break;
        }
    }
    let base = query.slot * MAX_HITS_PER_QUERY;
    if (base + slot < arrayLength(&query_hits)) {
        query_hits[base + slot] = QueryHit(body.body_id, body.generation, distance, 0u, point, 0.0, normal, 0.0);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) invocation_id: vec3u,
) {
    let query = queries[workgroup_id.x];
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
    while (large_index < large_count[0]) {
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
        let body = bodies[collider_slot / 4u];
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
            let static_target = world_collider(body, collider);
            let distance = sweep_convex(query, static_target, &normal);
            if (distance < NO_HIT) {
                let point = query.origin + normalize(query.direction) * distance;
                hit = ShapeHit(distance, point, normal);
            }
        } else if (query.kind == QUERY_SPHERE) {
            let separation = overlap_hit(query, body, collider, &normal);
            if (separation < NO_HIT) {
                let point = world_collider(body, collider).center;
                hit = ShapeHit(separation, point, normal);
            }
        } else {
            let world = world_collider(body, collider);
            let probe = query_shape_world(query, query.origin);
            var simplex: array<SimplexPoint, 4>;
            var count = 0u;
            let closest = convex_closest(probe, world, &simplex, &count);
            if (closest.penetrating) {
                hit = ShapeHit(0.0, (closest.point_a + closest.point_b) * 0.5, closest.normal);
            }
        }
        if (hit.distance < NO_HIT) {
            emit_hit(query, body, collider, hit.distance, hit.point, hit.normal);
        }
        candidate_index = candidate_index + WORKGROUP_SIZE;
    }
}
