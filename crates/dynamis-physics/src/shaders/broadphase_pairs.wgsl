@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(2) var<storage, read_write> pairs: array<Pair>;
@group(0) @binding(3) var<storage, read_write> pair_count: atomic<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let n = params.body_count;
    if (n < 2u) {
        return;
    }
    let total = n * (n - 1u) / 2u;
    let k = gid.x;
    if (k >= total) {
        return;
    }
    var i = u32((sqrt(f32(8.0 * f32(k) + 1.0)) + 1.0) / 2.0);
    while (i * (i - 1u) / 2u > k) {
        i = i - 1u;
    }
    while ((i + 1u) * i / 2u <= k) {
        i = i + 1u;
    }
    let j = k - (i * (i - 1u) / 2u);
    let first = aabbs[j];
    let second = aabbs[i];
    if (first.max.x >= second.min.x && first.min.x <= second.max.x &&
        first.max.y >= second.min.y && first.min.y <= second.max.y &&
        first.max.z >= second.min.z && first.min.z <= second.max.z) {
        let slot = atomicAdd(&pair_count, 1u);
        if (slot < arrayLength(&pairs)) {
            pairs[slot] = Pair(j, i);
        }
    }
}
