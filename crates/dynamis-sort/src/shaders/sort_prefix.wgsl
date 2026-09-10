@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>, 256>;
@group(0) @binding(1) var<storage, read_write> block_histogram: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> block_prefix: array<u32>;
@group(0) @binding(3) var<storage, read> count_holder: array<u32>;

const BIN_COUNT: u32 = 256u;
const TILE: u32 = 256u;

var<workgroup> partial: array<u32, TILE>;
var<workgroup> base: u32;

fn scan(lane: u32) -> u32 {
    var value = partial[lane];
    var stride = 1u;
    for (var step = 0u; step < 8u; step = step + 1u) {
        var source = 0u;
        if (lane >= stride) {
            source = partial[lane - stride];
        }
        workgroupBarrier();
        value = value + source;
        partial[lane] = value;
        workgroupBarrier();
        stride = stride * 2u;
    }
    return value;
}

@compute @workgroup_size(TILE)
fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let bin = wgid.x;
    let reserved = (arrayLength(&block_histogram) / BIN_COUNT) * TILE;
    let blocks = (min(count_holder[0], reserved) + TILE - 1u) / TILE;

    partial[lid.x] = select(0u, atomicLoad(&histogram[lid.x]), lid.x < bin);
    workgroupBarrier();
    let forward = scan(lid.x);
    if (lid.x == TILE - 1u) {
        base = forward;
    }
    workgroupBarrier();
    let offset = base;

    var owned = 0u;
    for (var block = lid.x; block < blocks; block = block + TILE) {
        owned = owned + atomicLoad(&block_histogram[block * BIN_COUNT + bin]);
    }
    partial[lid.x] = owned;
    workgroupBarrier();
    var running = scan(lid.x) - owned + offset;
    workgroupBarrier();
    for (var block = lid.x; block < blocks; block = block + TILE) {
        let held = atomicLoad(&block_histogram[block * BIN_COUNT + bin]);
        block_prefix[block * BIN_COUNT + bin] = running;
        atomicStore(&block_histogram[block * BIN_COUNT + bin], 0u);
        running = running + held;
    }
}
