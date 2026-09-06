@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(2) var<storage, read_write> entry_keys_hi: array<u32>;
@group(0) @binding(3) var<storage, read_write> entry_keys_lo: array<u32>;
@group(0) @binding(4) var<storage, read_write> entry_count: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> large_bodies: array<u32>;
@group(0) @binding(6) var<storage, read_write> large_count: atomic<u32>;

fn cell_hash(coord: vec3i) -> u32 {
    let x = u32(coord.x) * 0x9E3779B9u;
    let y = u32(coord.y) * 0x85EBCA77u;
    let z = u32(coord.z) * 0xC2B2AE3Du;
    return x ^ y ^ z ^ (x << 7u) ^ (y >> 3u) ^ (z << 11u);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.body_count) {
        return;
    }
    let aabb = aabbs[index];
    let cell_size = params.grid_cell_size;
    let min_cell = vec3i(floor(aabb.min / cell_size));
    let max_cell = vec3i(floor(aabb.max / cell_size));
    let span = max_cell - min_cell + vec3i(1);
    let cell_count = span.x * span.y * span.z;
    if (cell_count > i32(params.max_cells_per_body)) {
        let slot = atomicAdd(&large_count, 1u);
        if (slot < arrayLength(&large_bodies)) {
            large_bodies[slot] = index;
        }
        return;
    }
    for (var dx = min_cell.x; dx <= max_cell.x; dx = dx + 1) {
        for (var dy = min_cell.y; dy <= max_cell.y; dy = dy + 1) {
            for (var dz = min_cell.z; dz <= max_cell.z; dz = dz + 1) {
                let slot = atomicAdd(&entry_count, 1u);
                if (slot < arrayLength(&entry_keys_hi)) {
                    entry_keys_hi[slot] = cell_hash(vec3i(dx, dy, dz));
                    entry_keys_lo[slot] = index;
                }
            }
        }
    }
}
