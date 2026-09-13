@group(0) @binding(0) var<storage, read> block_sums: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> pair_count: array<atomic<u32>>;

var<workgroup> scratch: array<u32, 256>;

const BLOCK: u32 = 256u;

@compute @workgroup_size(256u)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let index = lid.x;
    let live_lanes = min(atomicLoad(&pair_count[0]), arrayLength(&block_sums) * BLOCK);
    let blocks = (live_lanes + BLOCK - 1u) / BLOCK;
    let per = (blocks + 255u) / 256u;
    var local = 0u;
    for (var offset = 0u; offset < per; offset = offset + 1u) {
        let block = index * per + offset;
        if (block >= blocks) {
            break;
        }
        local = local + block_sums[block];
    }
    scratch[index] = local;
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
    var group_base = value - local;
    for (var offset = 0u; offset < per; offset = offset + 1u) {
        let block = index * per + offset;
        if (block >= blocks) {
            break;
        }
        let count = block_sums[block];
        block_offsets[block] = group_base;
        group_base = group_base + count;
        if (block == blocks - 1u) {
            atomicStore(&contact_count[0], group_base);
        }
    }
}
