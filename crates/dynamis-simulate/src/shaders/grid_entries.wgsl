@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(2) var<storage, read_write> entry_cells: array<u32>;
@group(0) @binding(3) var<storage, read_write> entry_colliders: array<u32>;
@group(0) @binding(4) var<storage, read_write> entry_count: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> large_bodies: array<u32>;
@group(0) @binding(6) var<storage, read_write> large_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> spillover: array<atomic<u32>>;

fn cell_hash(coord: vec3i) -> u32 {
    let x = u32(coord.x) * 0x9E3779B9u;
    let y = u32(coord.y) * 0x85EBCA77u;
    let z = u32(coord.z) * 0xC2B2AE3Du;
    return x ^ y ^ z ^ (x << 7u) ^ (y >> 3u) ^ (z << 11u);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.body_count) {
        return;
    }
    let cell_size = params.grid_cell_size;
    for (var i = 0u; i < MAX_COLLIDERS_PER_BODY; i = i + 1u) {
        let collider_index = index * MAX_COLLIDERS_PER_BODY + i;
        let aabb = aabbs[collider_index];
        if (aabb.min.x > aabb.max.x) {
            continue;
        }
        let min_cell = vec3i(floor(aabb.min / cell_size));
        let max_cell = vec3i(floor(aabb.max / cell_size));
        let span = max_cell - min_cell + vec3i(1);
        let cell_count = span.x * span.y * span.z;
        if (cell_count > i32(params.max_cells_per_collider) || span.x > 8192 || span.y > 8192 || span.z > 8192) {
            let slot = atomicAdd(&large_count[0], 1u);
            if (slot < arrayLength(&large_bodies)) {
                large_bodies[slot] = collider_index;
            }
            continue;
        }
        for (var dx = min_cell.x; dx <= max_cell.x; dx = dx + 1) {
            for (var dy = min_cell.y; dy <= max_cell.y; dy = dy + 1) {
                for (var dz = min_cell.z; dz <= max_cell.z; dz = dz + 1) {
                    let slot = atomicAdd(&entry_count[0], 1u);
                    if (slot < arrayLength(&entry_cells)) {
                        entry_cells[slot] = cell_hash(vec3i(dx, dy, dz));
                        entry_colliders[slot] = collider_index;
                    } else {
                        atomicAdd(&spillover[0], 1u);
                    }
                }
            }
        }
    }
}
