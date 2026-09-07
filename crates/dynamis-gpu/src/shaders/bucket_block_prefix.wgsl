@group(0) @binding(0) var<storage, read_write> block_histogram: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> block_prefix: array<atomic<u32>>;

const BUCKET_COUNT: u32 = __BUCKETS__u;

@compute @workgroup_size(256u)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let digit = lid.x;
    let blocks = arrayLength(&block_histogram) / BUCKET_COUNT;
    let per = (BUCKET_COUNT + 255u) / 256u;
    for (var offset = 0u; offset < per; offset = offset + 1u) {
        let bucket = digit * per + offset;
        if (bucket >= BUCKET_COUNT) {
            break;
        }
        var prefix = 0u;
        for (var block = 0u; block < blocks; block = block + 1u) {
            let count = atomicLoad(&block_histogram[block * BUCKET_COUNT + bucket]);
            atomicStore(&block_prefix[block * BUCKET_COUNT + bucket], prefix);
            atomicStore(&block_histogram[block * BUCKET_COUNT + bucket], 0u);
            prefix = prefix + count;
        }
    }
}
