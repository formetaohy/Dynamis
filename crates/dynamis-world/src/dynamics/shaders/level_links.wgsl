@group(0) @binding(3) var<uniform> params: StepParams;
@group(0) @binding(4) var<storage, read_write> pair_major: array<u32>;
@group(0) @binding(5) var<storage, read_write> pair_minor: array<u32>;
@group(0) @binding(6) var<storage, read_write> pair_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> spillover: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(9) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(10) var<storage, read> body_activity: array<u32>;
@group(0) @binding(11) var<storage, read_write> levels: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> active_coarse: array<atomic<u32>>;

fn emit_pair(first: u32, second: u32) {
    let owner = collider_owners[first];
    if (first == second || owner == collider_owners[second]) {
        return;
    }
    let a = min(first, second);
    let b = max(first, second);
    let slot = atomicAdd(&pair_count[0], 1u);
    if (slot < arrayLength(&pair_minor)) {
        pair_minor[slot] = b;
        pair_major[slot] = a;
    } else {
        atomicAdd(&spillover[0], 1u);
    }
}

fn link_level(collider: u32, aabb: Aabb, level: u32, awake: bool, live: u32) {
    let cell_size = level_cell_size(level, params.grid_cell_size);
    let min_cell = vec3i(floor(aabb.min / cell_size));
    let max_cell = vec3i(floor(aabb.max / cell_size));
    for (var dx = min_cell.x; dx <= max_cell.x; dx = dx + 1) {
        for (var dy = min_cell.y; dy <= max_cell.y; dy = dy + 1) {
            for (var dz = min_cell.z; dz <= max_cell.z; dz = dz + 1) {
                let range = entry_bounds(live, cell_key(level, vec3i(dx, dy, dz)));
                for (var entry = range.x; entry < range.y; entry = entry + 1u) {
                    let other = entry_colliders[entry];
                    if (other == collider) {
                        continue;
                    }
                    let other_owner = collider_owners[other];
                    if (other_owner == NO_BODY) {
                        continue;
                    }
                    if (!awake && body_activity[other_owner] == 0u) {
                        continue;
                    }
                    if (!aabb_overlaps(aabb, aabbs[other])) {
                        continue;
                    }
                    emit_pair(collider, other);
                }
            }
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let collider = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (collider >= params.collider_count) {
        return;
    }
    let owner = collider_owners[collider];
    if (owner == NO_BODY) {
        return;
    }
    let aabb = aabbs[collider];
    let level = shape_levels(aabb, params.grid_cell_size);
    let awake = body_activity[owner] != 0u;
    if (!awake && atomicLoad(&active_coarse[0]) == 0u) {
        return;
    }
    let live = entry_live();
    var coarser = atomicLoad(&levels[0]) & ~((1u << min(level + 1u, 31u)) - 1u);
    while (coarser != 0u) {
        let link = countTrailingZeros(coarser);
        coarser = coarser & (coarser - 1u);
        link_level(collider, aabb, link, awake, live);
    }
}
