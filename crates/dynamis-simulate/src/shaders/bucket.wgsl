@group(0) @binding(0) var<storage, read> islands: array<u32>;
@group(0) @binding(1) var<storage, read> source_keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read> source_keys_lo: array<u32>;
@group(0) @binding(3) var<storage, write> bucket_keys_hi: array<u32>;
@group(0) @binding(4) var<storage, write> bucket_keys_lo: array<u32>;
@group(0) @binding(5) var<storage, write> bucket_values: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params_count) {
        return;
    }
    let root = islands[source_keys_hi[index] / MAX_COLLIDERS_PER_BODY];
    bucket_keys_hi[index] = root;
    bucket_keys_lo[index] = source_keys_lo[index];
    bucket_values[index] = index;
}
