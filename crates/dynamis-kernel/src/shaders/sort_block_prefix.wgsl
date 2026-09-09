@group(0) @binding(0) var<storage, read_write> block_histogram: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> block_prefix: array<u32>;

const BIN_COUNT: u32 = 256u;
const TILE: u32 = 256u;

var<workgroup> partial: array<u32, TILE>;

@compute @workgroup_size(TILE)
fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let bin = wgid.x;
    let blocks = arrayLength(&block_histogram) / BIN_COUNT;
    var total = 0u;
    for (var block = lid.x; block < blocks; block = block + TILE) {
        total = total + atomicLoad(&block_histogram[block * BIN_COUNT + bin]);
    }
    partial[lid.x] = total;
    workgroupBarrier();
    var value = total;
    var stride = 1u;
    for (var step = 0u; step < 8u; step = step + 1u) {
        var source = 0u;
        if (lid.x >= stride) {
            source = partial[lid.x - stride];
        }
        workgroupBarrier();
        value = value + source;
        partial[lid.x] = value;
        workgroupBarrier();
        stride = stride * 2u;
    }
    var running = value - total;
    workgroupBarrier();
    for (var block = lid.x; block < blocks; block = block + TILE) {
        let count = atomicLoad(&block_histogram[block * BIN_COUNT + bin]);
        block_prefix[block * BIN_COUNT + bin] = running;
        atomicStore(&block_histogram[block * BIN_COUNT + bin], 0u);
        running = running + count;
    }
}
