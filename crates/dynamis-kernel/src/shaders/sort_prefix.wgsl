@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>, 256>;
@group(0) @binding(1) var<storage, read_write> cursor: array<atomic<u32>, 256>;
@group(0) @binding(2) var<storage, read_write> block_histogram: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> block_prefix: array<atomic<u32>>;

const BIN_COUNT: u32 = 256u;

var<workgroup> scratch: array<u32, 256>;

@compute @workgroup_size(256u)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let index = lid.x;
    scratch[index] = atomicLoad(&histogram[index]);
    workgroupBarrier();
    var value = scratch[index];
    var stride = 1u;
    for (var step = 0u; step < 8u; step = step + 1u) {
        var source = 0u;
        if (index >= stride) {
            source = scratch[index - stride];
        }
        workgroupBarrier();
        value = value + source;
        scratch[index] = value;
        workgroupBarrier();
        stride = stride * 2u;
    }
    let exclusive = value - atomicLoad(&histogram[index]);
    atomicStore(&cursor[index], exclusive);
    atomicStore(&histogram[index], 0u);
    let blocks = arrayLength(&block_histogram) / BIN_COUNT;
    var prefix = 0u;
    for (var block = 0u; block < blocks; block = block + 1u) {
        let count = atomicLoad(&block_histogram[block * BIN_COUNT + index]);
        atomicStore(&block_prefix[block * BIN_COUNT + index], prefix);
        atomicStore(&block_histogram[block * BIN_COUNT + index], 0u);
        prefix = prefix + count;
    }
}
