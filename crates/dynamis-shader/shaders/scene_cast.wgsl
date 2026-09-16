@group(0) @binding(4) var<storage, read_write> queries: array<Query>;
@group(0) @binding(5) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(6) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(7) var<storage, read> colliders: array<Collider>;
@group(0) @binding(8) var<storage, read_write> query_hits: array<QueryHit>;
@group(0) @binding(9) var<uniform> params: StepParams;
@group(0) @binding(10) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(11) var<storage, read> particles: array<SoftParticle>;
@group(0) @binding(12) var<storage, read> soft_bodies: array<SoftBody>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

var<workgroup> candidates: array<u32, QUERY_CANDIDATES>;
var<workgroup> candidate_count: atomic<u32>;
var<workgroup> overflow_flag: atomic<u32>;
var<workgroup> pick_key: atomic<u32>;
var<workgroup> pick_slot: atomic<u32>;
var<workgroup> hit_total: atomic<u32>;
var<workgroup> threshold_key: u32;
var<workgroup> threshold_slot: u32;

fn sort_width(candidate_total: u32) -> u32 {
    var width = WORKGROUP_SIZE;
    loop {
        if (width >= candidate_total || width >= QUERY_CANDIDATES) {
            break;
        }
        width = width * 2u;
    }
    return min(width, QUERY_CANDIDATES);
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
    return scene_target_of(kind, entry_index(info));
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
    if (slot >= QUERY_CANDIDATES) {
        atomicStore(&overflow_flag, 1u);
        return;
    }
    candidates[slot] = candidate;
}

fn collect_whole_slice(slice: GridSlice, query_box: Aabb, invocation: u32, query: Query) {
    var entry = slice.first + invocation;
    while (entry < slice.end && atomicLoad(&candidate_count) <= QUERY_CANDIDATES) {
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
    var hit = shape_ray(world_collider(body.state, collider), query.origin, direction, query.extent);
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
    var hit = shape_ray(world, query.origin, direction, query.extent);
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
        let hit = shape_sweep(query_shape_world(query, query.origin), world, query.origin, query.direction, query.extent);
        if (hit.distance < NO_HIT) {
            return hit;
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

fn eligible(key: u32, slot: u32) -> bool {
    return key > threshold_key || (key == threshold_key && slot > threshold_slot);
}

fn nearer(key: u32, slot: u32, key_now: u32, slot_now: u32) -> bool {
    return key < key_now || (key == key_now && slot < slot_now);
}

fn resolve_hit(query: Query, body: Body, collider: Collider) -> ShapeHit {
    if (query.kind == QUERY_RAY) {
        return ray_hit(query, body, collider);
    }
    var normal = vec3f(0.0);
    let is_world_geom = shape_world_geometry(collider.kind);
    if (query.kind == QUERY_SWEEP) {
        let static_target = world_collider(body.state, collider);
        let hit = shape_sweep(query_shape_world(query, query.origin), static_target, query.origin, query.direction, query.extent);
        if (hit.distance < NO_HIT) {
            return hit;
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
    let capacity = min(query.max_hits, QUERY_CANDIDATES);
    let base = query.hit_base;
    if (invocation_id.x == 0u) {
        threshold_key = 0u;
        threshold_slot = NO_CANDIDATE;
        atomicStore(&hit_total, 0u);
    }
    workgroupBarrier();
    atomicStore(&candidate_count, 0u);
    atomicStore(&overflow_flag, 0u);
    workgroupBarrier();
    let query_box = query_world_aabb(query);
    collect_candidates(query_box, invocation_id.x, query);
    workgroupBarrier();
    let candidate_total = min(atomicLoad(&candidate_count), QUERY_CANDIDATES);
    let width = sort_width(candidate_total);
    for (var index = invocation_id.x; index < width; index = index + WORKGROUP_SIZE) {
        if (index >= candidate_total) {
            candidates[index] = NO_CANDIDATE;
        }
    }
    workgroupBarrier();
    bitonic_sort(invocation_id.x, width);
    workgroupBarrier();
    var emitted = 0u;
    var round = 0u;
    loop {
        if (emitted >= capacity) {
            break;
        }
        var mine_key = NO_KEY;
        var mine_slot = NO_CANDIDATE;
        var mine_hit = no_hit();
        var resolved = 0u;
        var candidate_index = invocation_id.x;
        while (candidate_index < candidate_total) {
            let candidate = candidates[candidate_index];
            let duplicate = candidate_index > 0u && candidate == candidates[candidate_index - 1u];
            if (!duplicate && candidate_passes_filters(query, candidate)) {
                var normal = vec3f(0.0);
                let hit = resolve_candidate_hit(query, candidate, &normal);
                if (hit.distance < NO_HIT) {
                    if (round == 0u) {
                        resolved = resolved + 1u;
                    }
                    let key = hit_key(hit.distance);
                    if (eligible(key, candidate) && nearer(key, candidate, mine_key, mine_slot)) {
                        mine_key = key;
                        mine_slot = candidate;
                        mine_hit = hit;
                    }
                }
            }
            candidate_index = candidate_index + WORKGROUP_SIZE;
        }
        if (round == 0u) {
            atomicAdd(&hit_total, resolved);
        }
        if (invocation_id.x == 0u) {
            atomicStore(&pick_key, NO_KEY);
            atomicStore(&pick_slot, NO_CANDIDATE);
        }
        workgroupBarrier();
        atomicMin(&pick_key, mine_key);
        workgroupBarrier();
        let chosen_key = atomicLoad(&pick_key);
        if (mine_key == chosen_key) {
            atomicMin(&pick_slot, mine_slot);
        }
        workgroupBarrier();
        let chosen_slot = atomicLoad(&pick_slot);
        if (mine_key == chosen_key && mine_slot == chosen_slot && chosen_key != NO_KEY) {
            let owner = candidate_owner(mine_slot);
            query_hits[base + emitted] = QueryHit(owner.x, owner.y, mine_hit.distance, mine_slot, mine_hit.point, mine_hit.triangle, mine_hit.normal, candidate_surface(mine_slot, mine_hit.triangle));
            threshold_key = chosen_key;
            threshold_slot = chosen_slot;
        }
        workgroupBarrier();
        let found = chosen_key != NO_KEY;
        emitted = emitted + select(0u, 1u, found);
        round = round + 1u;
        if (!found) {
            break;
        }
    }
    if (invocation_id.x == 0u) {
        let spilling = atomicLoad(&overflow_flag) != 0u;
        let total_hits = atomicLoad(&hit_total);
        queries[batch].count = emitted;
        queries[batch].overflow = select(0u, 1u, spilling || emitted < total_hits);
    }
}
