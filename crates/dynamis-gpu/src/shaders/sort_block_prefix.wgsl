const BIN_COUNT: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> block_histogram: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> block_prefix: array<atomic<u32>>;

@compute @workgroup_size(BIN_COUNT)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let digit = lid.x;
    let blocks = arrayLength(&block_histogram) / BIN_COUNT;
    var prefix = 0u;
    for (var block = 0u; block < blocks; block = block + 1u) {
        let count = atomicLoad(&block_histogram[block * BIN_COUNT + digit]);
        atomicStore(&block_prefix[block * BIN_COUNT + digit], prefix);
        atomicStore(&block_histogram[block * BIN_COUNT + digit], 0u);
        prefix = prefix + count;
    }
}
