@group(0) @binding(4) var<storage, read> queries: array<Query>;
@group(0) @binding(5) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(6) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(7) var<storage, read> colliders: array<Collider>;
@group(0) @binding(8) var<storage, read_write> query_results: array<QueryResult>;
@group(0) @binding(9) var<uniform> params: StepParams;
@group(0) @binding(10) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(11) var<storage, read> particles: array<SoftParticle>;
@group(0) @binding(12) var<storage, read> soft_bodies: array<SoftBody>;

const CANDIDATES_PER_QUERY: u32 = 4096u;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

var<workgroup> candidates: array<u32, CANDIDATES_PER_QUERY>;
var<workgroup> candidate_count: atomic<u32>;
var<workgroup> overflow_flag: atomic<u32>;
var<workgroup> valid_count: atomic<u32>;
var<workgroup> pick_key: atomic<u32>;
var<workgroup> pick_slot: atomic<u32>;

fn sort_width(candidate_total: u32) -> u32 {
    var width = WORKGROUP_SIZE;
    loop {
        if (width >= candidate_total || width >= CANDIDATES_PER_QUERY) {
            break;
        }
        width = width * 2u;
    }
    return min(width, CANDIDATES_PER_QUERY);
}

fn bitonic_sort(local_invocation: u32, width: u32) {
    var k = 2u;
    loop {
        if (k > width) {
            break;
        }
        var j = k / 2u;
        loop {
            if (j == 0u) {
                break;
            }
            var index = local_invocation;
            while (index < width) {
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

fn scene_target(entry: u32) -> u32 {
    let info = entry_info(entry_node(entry));
    let kind = entry_kind(info);
    if (kind != ENTRY_KIND_COLLIDER && kind != ENTRY_KIND_PARTICLE) {
        return NO_SLOT;
    }
    return (kind << ENTRY_KIND_SHIFT) | entry_index(info);
}

fn candidate_kind(candidate: u32) -> u32 {
    return candidate >> ENTRY_KIND_SHIFT;
}

fn candidate_index(candidate: u32) -> u32 {
    return candidate & ENTRY_INDEX_MASK;
}

fn candidate_passes(query: Query, candidate: u32) -> bool {
    return (query.filters.targets & (1u << candidate_kind(candidate))) != 0u;
}

fn collect_entry(entry: u32, query_box: Aabb, whole: bool, query: Query) {
    let candidate = scene_target(entry);
    if (candidate == NO_SLOT || !candidate_passes(query, candidate)) {
        return;
    }
    if (whole && !aabb_overlaps(entry_box(entry), query_box)) {
        return;
    }
    let slot = atomicAdd(&candidate_count, 1u);
    if (slot >= CANDIDATES_PER_QUERY) {
        atomicStore(&overflow_flag, 1u);
        return;
    }
    candidates[slot] = candidate;
}

fn collect_whole_slice(slice: GridSlice, query_box: Aabb, invocation: u32, query: Query) {
    var entry = slice.first + invocation;
    while (entry < slice.end && atomicLoad(&candidate_count) <= CANDIDATES_PER_QUERY) {
        collect_entry(entry, query_box, true, query);
        entry = entry + WORKGROUP_SIZE;
    }
}

fn collect_cell_slice(slice: GridSlice, query_box: Aabb, query: Query) {
    for (var entry = slice.first; entry < slice.end; entry = entry + 1u) {
        collect_entry(entry, query_box, false, query);
    }
}

fn collect_candidates(query_box: Aabb, invocation: u32, query: Query) {
    let whole = grid_whole_slices(query_box);
    for (var index = 0u; index < whole; index = index + 1u) {
        collect_whole_slice(grid_whole_slice(query_box, index), query_box, invocation, query);
    }
    let cells = grid_cell_slices(query_box);
    var index = invocation;
    while (index < cells) {
        collect_cell_slice(grid_cell_slice(query_box, index), query_box, query);
        index = index + WORKGROUP_SIZE;
    }
}

fn ray_hit(query: Query, body: Body, collider: Collider) -> ShapeHit {
    let direction = normalize(query.direction);
    let world = world_collider(body.state, collider);
    var hit = no_hit();
    if (collider.kind == SHAPE_PLANE) {
        let n = plane_normal(world);
        let denom = dot(direction, n);
        if (abs(denom) < 1e-8) {
            return no_hit();
        }
        let t = dot(world.center - query.origin, n) / denom;
        if (t < 0.0 || t > query.extent) {
            return no_hit();
        }
        hit = ShapeHit(t, query.origin + direction * t, n, NO_TRIANGLE);
    } else if (collider.kind == SHAPE_MESH || collider.kind == SHAPE_HEIGHTFIELD || collider.kind == SHAPE_HULL) {
        hit = ray_scene(world, query.origin, direction, query.extent);
    } else {
        hit = ray_scaled_shape(world, query.origin, direction, query.extent, 0.0);
    }
    if (hit.distance == NO_HIT) {
        return hit;
    }
    if (dot(hit.normal, direction) > 0.0) {
        hit.normal = -hit.normal;
    }
    return hit;
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
    let moving = query_shape_world(query, start);
    if (static_target.kind == SHAPE_PLANE) {
        let n = plane_normal(static_target);
        let travel = dot(direction, n);
        if (travel >= 0.0) {
            return NO_HIT;
        }
        let offset = dot(start - static_target.center, n);
        if (offset <= 0.0) {
            return NO_HIT;
        }
        let extent = dot(support(moving, -n) - moving.center, -n);
        let time = max((offset - extent) / (-travel), 0.0);
        if (time > query.extent) {
            return NO_HIT;
        }
        *out_normal = n;
        return time;
    }
    let hit = scene_convex_sweep(static_target, moving, start, direction, query.extent);
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
    var normal = sign_normalize(query.origin - static_target.center);
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
        normal = -closest.normal;
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
        *out_normal = -hit.normal;
        return -query.extent + max(hit.distance + query.extent, 0.0);
    }
    if (closest.distance <= query.extent) {
        *out_normal = -closest.normal;
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
    return closest.distance;
}

fn body_passes(body: Body, collider: Collider, query: Query) -> bool {
    if (query.filters.exclude_id != NO_BODY && body.state.body_id == query.filters.exclude_id && body.state.generation == query.filters.exclude_generation) {
        return false;
    }
    if (query.filters.include_id != NO_BODY && (body.state.body_id != query.filters.include_id || body.state.generation != query.filters.include_generation)) {
        return false;
    }
    if ((query.filters.flags & FILTER_IGNORE_SENSORS) != 0u && (collider.flags & COLLIDER_SENSOR) != 0u) {
        return false;
    }
    if ((query.filters.flags & FILTER_IGNORE_SLEEPING) != 0u && body.state.sleeping != 0u) {
        return false;
    }
    if ((query.filters.flags & FILTER_IGNORE_STATIC) != 0u && body_is_static(body)) {
        return false;
    }
    if ((query.filters.flags & FILTER_IGNORE_KINEMATIC) != 0u && body_is_kinematic(body)) {
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

struct SoftFilter {
    sleeping: u32,
    group: u32,
    mask: u32,
}

fn soft_filter(owner: u32) -> SoftFilter {
    return SoftFilter(
        soft_bodies[owner].sleeping,
        soft_bodies[owner].collision_group,
        soft_bodies[owner].collision_mask,
    );
}

fn soft_filter_query(query: Query, soft: SoftFilter) -> bool {
    if (query.filters.group == 0u) {
        return true;
    }
    return filters_intersect(
        vec2u(query.filters.group, query.filters.mask),
        vec2u(soft.group, soft.mask),
    );
}

fn particle_passes(particle: SoftParticle, soft: SoftFilter, query: Query) -> bool {
    if (query.filters.exclude_soft_id != NO_BODY
        && particle.owner == query.filters.exclude_soft_id
        && particle.generation == query.filters.exclude_soft_generation) {
        return false;
    }
    if (query.filters.include_soft_id != NO_BODY
        && (particle.owner != query.filters.include_soft_id || particle.generation != query.filters.include_soft_generation)) {
        return false;
    }
    if ((query.filters.flags & FILTER_IGNORE_SLEEPING) != 0u && soft.sleeping != 0u) {
        return false;
    }
    return soft_filter_query(query, soft);
}

fn particle_shape(center: vec3f, radius: f32) -> WorldShape {
    var world: WorldShape;
    world.kind = SHAPE_SPHERE;
    world.radius = radius;
    world.half_height = 0.0;
    world.center = center;
    world.half_extents = vec3f(0.0);
    world.rotation = vec4f(0.0, 0.0, 0.0, 1.0);
    world.source = 0u;
    world.scale = vec3f(1.0);
    return world;
}

fn ray_particle(query: Query, world: WorldShape) -> ShapeHit {
    let direction = normalize(query.direction);
    var hit = ray_scaled_shape(world, query.origin, direction, query.extent, 0.0);
    if (hit.distance == NO_HIT) {
        return hit;
    }
    if (dot(hit.normal, direction) > 0.0) {
        hit.normal = -hit.normal;
    }
    return hit;
}

fn particle_hit(query: Query, center: vec3f, radius: f32, out_normal: ptr<function, vec3f>) -> ShapeHit {
    let world = particle_shape(center, radius);
    if (query.kind == QUERY_RAY) {
        return ray_particle(query, world);
    }
    if (query.kind == QUERY_SWEEP) {
        let distance = sweep_convex(query, world, out_normal);
        if (distance < NO_HIT) {
            return ShapeHit(distance, query.origin + normalize(query.direction) * distance, *out_normal, NO_TRIANGLE);
        }
        return no_hit();
    }
    var probe = query_shape_world(query, query.origin);
    probe.radius = 0.0;
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    let closest = convex_closest(probe, world, &simplex, &count);
    if (closest.penetrating) {
        let hit = convex_hit(query_shape_world(query, query.origin), world);
        *out_normal = -hit.normal;
        return ShapeHit(-query.extent + max(hit.distance + query.extent, 0.0), center, *out_normal, NO_TRIANGLE);
    }
    if (closest.distance <= query.extent) {
        *out_normal = -closest.normal;
        return ShapeHit(closest.distance - query.extent, center, *out_normal, NO_TRIANGLE);
    }
    return no_hit();
}

fn resolve_candidate_hit(query: Query, candidate: u32, out_normal: ptr<function, vec3f>) -> ShapeHit {
    if (candidate_kind(candidate) == ENTRY_KIND_PARTICLE) {
        let particle = particles[candidate_index(candidate)];
        return particle_hit(query, particle.position.xyz, particle.position.w, out_normal);
    }
    let body = load_body(collider_owners[candidate_index(candidate)]);
    return resolve_hit(query, body, colliders[candidate_index(candidate)]);
}

fn candidate_passes_filters(query: Query, candidate: u32) -> bool {
    if (candidate_kind(candidate) == ENTRY_KIND_PARTICLE) {
        let particle = particles[candidate_index(candidate)];
        return particle_passes(particle, soft_filter(particle.owner), query);
    }
    let body = load_body(collider_owners[candidate_index(candidate)]);
    return body_passes(body, colliders[candidate_index(candidate)], query);
}

fn candidate_owner(candidate: u32) -> vec2u {
    if (candidate_kind(candidate) == ENTRY_KIND_PARTICLE) {
        let particle = particles[candidate_index(candidate)];
        return vec2u(particle.owner, particle.generation);
    }
    let body = load_body(collider_owners[candidate_index(candidate)]);
    return vec2u(body.state.body_id, body.state.generation);
}

fn candidate_surface(candidate: u32, triangle: u32) -> u32 {
    if (candidate_kind(candidate) == ENTRY_KIND_PARTICLE) {
        return NO_SURFACE;
    }
    return triangle_surface_index(colliders[candidate_index(candidate)], triangle);
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

fn hit_key(distance: f32) -> u32 {
    let bits = bitcast<u32>(distance);
    if ((bits & 0x80000000u) != 0u) {
        return ~bits;
    }
    return bits | 0x80000000u;
}

fn keep_nearest(
    keys: ptr<function, array<u32, MAX_HITS_PER_QUERY>>,
    slots: ptr<function, array<u32, MAX_HITS_PER_QUERY>>,
    stored: ptr<function, u32>,
    capacity: u32,
    key: u32,
    slot: u32,
) {
    if (capacity == 0u || (*stored) >= capacity && key >= (*keys)[capacity - 1u]) {
        return;
    }
    var position = min(*stored, capacity - 1u);
    loop {
        if (position == 0u) {
            break;
        }
        let previous_key = (*keys)[position - 1u];
        let previous_slot = (*slots)[position - 1u];
        if (previous_key < key || (previous_key == key && previous_slot < slot)) {
            break;
        }
        (*keys)[position] = previous_key;
        (*slots)[position] = previous_slot;
        position = position - 1u;
    }
    (*keys)[position] = key;
    (*slots)[position] = slot;
    if ((*stored) < capacity) {
        (*stored) = (*stored) + 1u;
    }
}

fn drop_nearest(
    keys: ptr<function, array<u32, MAX_HITS_PER_QUERY>>,
    slots: ptr<function, array<u32, MAX_HITS_PER_QUERY>>,
    stored: ptr<function, u32>,
) {
    for (var index = 1u; index < (*stored); index = index + 1u) {
        (*keys)[index - 1u] = (*keys)[index];
        (*slots)[index - 1u] = (*slots)[index];
    }
    (*stored) = (*stored) - 1u;
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
            return ShapeHit(distance, query.origin + normalize(query.direction) * distance, normal, NO_TRIANGLE);
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
            return ShapeHit(separation, world_collider(body.state, collider).center, normal, NO_TRIANGLE);
        }
        return no_hit();
    }
    if (is_world_geom) {
        let separation = overlap_world_geom(query, body, collider, &normal);
        if (separation < NO_HIT) {
            return ShapeHit(separation, world_collider(body.state, collider).center, normal, NO_TRIANGLE);
        }
        return no_hit();
    }
    let world = world_collider(body.state, collider);
    let probe = query_shape_world(query, query.origin);
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    let closest = convex_closest(probe, world, &simplex, &count);
    if (closest.penetrating) {
        return ShapeHit(0.0, (closest.point_a + closest.point_b) * 0.5, closest.normal, NO_TRIANGLE);
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
    }
    workgroupBarrier();
    atomicStore(&candidate_count, 0u);
    atomicStore(&overflow_flag, 0u);
    workgroupBarrier();
    let query_box = query_world_aabb(query);
    collect_candidates(query_box, invocation_id.x, query);
    workgroupBarrier();
    if (invocation_id.x == 0u && atomicLoad(&overflow_flag) != 0u) {
        atomicStore(&query_results[batch].header.overflow, 1u);
    }
    let candidate_total = min(atomicLoad(&candidate_count), CANDIDATES_PER_QUERY);
    let width = sort_width(candidate_total);
    for (var index = invocation_id.x; index < width; index = index + WORKGROUP_SIZE) {
        if (index >= candidate_total) {
            candidates[index] = NO_CANDIDATE;
        }
    }
    workgroupBarrier();
    bitonic_sort(invocation_id.x, width);
    workgroupBarrier();
    let capacity = min(query.max_hits, MAX_HITS_PER_QUERY);
    var nearest_key: array<u32, MAX_HITS_PER_QUERY>;
    var nearest_slot: array<u32, MAX_HITS_PER_QUERY>;
    var stored = 0u;
    var resolved = 0u;
    var candidate_index = invocation_id.x;
    while (candidate_index < candidate_total) {
        let candidate = candidates[candidate_index];
        let duplicate = candidate_index > 0u && candidate == candidates[candidate_index - 1u];
        if (!duplicate && candidate_passes_filters(query, candidate)) {
            var normal = vec3f(0.0);
            let hit = resolve_candidate_hit(query, candidate, &normal);
            if (hit.distance < NO_HIT) {
                resolved = resolved + 1u;
                keep_nearest(&nearest_key, &nearest_slot, &stored, capacity, hit_key(hit.distance), candidate);
            }
        }
        candidate_index = candidate_index + WORKGROUP_SIZE;
    }
    if (invocation_id.x == 0u) {
        atomicStore(&valid_count, 0u);
    }
    workgroupBarrier();
    atomicAdd(&valid_count, resolved);
    workgroupBarrier();
    let total_hits = atomicLoad(&valid_count);
    let emitted_total = min(total_hits, capacity);
    var round = 0u;
    loop {
        if (round >= emitted_total) {
            break;
        }
        if (invocation_id.x == 0u) {
            atomicStore(&pick_key, NO_KEY);
            atomicStore(&pick_slot, NO_KEY);
        }
        workgroupBarrier();
        var head_key = NO_KEY;
        var head_slot = NO_KEY;
        if (stored > 0u) {
            head_key = nearest_key[0];
            head_slot = nearest_slot[0];
        }
        atomicMin(&pick_key, head_key);
        workgroupBarrier();
        let chosen_key = atomicLoad(&pick_key);
        if (head_key == chosen_key) {
            atomicMin(&pick_slot, head_slot);
        }
        workgroupBarrier();
        let chosen_slot = atomicLoad(&pick_slot);
        if (head_key == chosen_key && head_slot == chosen_slot) {
            var normal = vec3f(0.0);
            let hit = resolve_candidate_hit(query, chosen_slot, &normal);
            let owner = candidate_owner(chosen_slot);
            query_results[batch].hits[round] = QueryHit(owner.x, owner.y, hit.distance, chosen_slot, hit.point, hit.triangle, hit.normal, candidate_surface(chosen_slot, hit.triangle));
            drop_nearest(&nearest_key, &nearest_slot, &stored);
        }
        round = round + 1u;
        workgroupBarrier();
    }
    if (invocation_id.x == 0u) {
        let spilling = atomicLoad(&overflow_flag) != 0u;
        atomicStore(&query_results[batch].header.count, emitted_total);
        atomicStore(&query_results[batch].header.overflow, select(0u, 1u, spilling || emitted_total < total_hits));
    }
}
