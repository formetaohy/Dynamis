@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> large_bodies: array<u32>;
@group(0) @binding(2) var<storage, read_write> large_count: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> pair_keys_hi: array<u32>;
@group(0) @binding(4) var<storage, read_write> pair_keys_lo: array<u32>;
@group(0) @binding(5) var<storage, read_write> pair_count: atomic<u32>;
@group(0) @binding(6) var<storage, read> colliders: array<Collider>;

fn emit_pair(first: u32, second: u32) {
    if (first == second) {
        return;
    }
    let a = min(first, second);
    let b = max(first, second);
    let slot = atomicAdd(&pair_count, 1u);
    if (slot < arrayLength(&pair_keys_lo)) {
        pair_keys_lo[slot] = b;
        pair_keys_hi[slot] = a;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.body_count) {
        return;
    }
    let large_count = atomicLoad(&large_count);
    if (large_count == 0u) {
        return;
    }
    for (var i = 0u; i < MAX_COLLIDERS_PER_BODY; i = i + 1u) {
        let collider_index = index * MAX_COLLIDERS_PER_BODY + i;
        if (colliders[collider_index].kind == SHAPE_NONE) {
            continue;
        }
        for (var large = 0u; large < large_count; large = large + 1u) {
            emit_pair(large_bodies[large], collider_index);
        }
    }
}
