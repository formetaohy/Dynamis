@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, __BUCKETS__>;
@group(0) @binding(2) var<storage, read_write> block_histogram: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> count_holder: array<u32>;

const BUCKET_COUNT: u32 = __BUCKETS__u;
const BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(BLOCK_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= count_holder[0]) {
        return;
    }
    let key = keys[index];
    atomicAdd(&histogram[key], 1u);
    atomicAdd(&block_histogram[(index / BLOCK_SIZE) * BUCKET_COUNT + key], 1u);
}
