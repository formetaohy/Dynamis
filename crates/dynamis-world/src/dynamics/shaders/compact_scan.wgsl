@group(0) @binding(0) var<storage, read> valid: array<u32>;
@group(0) @binding(1) var<storage, read_write> ranks: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(3) var<storage, read_write> count_holder: array<atomic<u32>>;

const BLOCK_SIZE: u32 = 256u;

var<workgroup> prefix: array<u32, BLOCK_SIZE>;
var<workgroup> flags: array<u32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let live = min(atomicLoad(&count_holder[0]), arrayLength(&valid));
    let tiles = groups.x * groups.y;
    for (var tile = wgid.y * WORKGROUPS_PER_ROW + wgid.x; tile * BLOCK_SIZE < live; tile = tile + tiles) {
        let index = tile * BLOCK_SIZE + lid.x;
        let flag = select(0u, 1u, index < live && valid[index] != 0u);
        flags[lid.x] = flag;
        workgroupBarrier();
        var value = flag;
        var stride = 1u;
        for (var step = 0u; step < 8u; step = step + 1u) {
            var source = 0u;
            if (lid.x >= stride) {
                source = flags[lid.x - stride];
            }
            workgroupBarrier();
            value = value + source;
            flags[lid.x] = value;
            workgroupBarrier();
            stride = stride * 2u;
        }
        prefix[lid.x] = value - flag;
        workgroupBarrier();
        ranks[index] = select(0xFFFFFFFFu, prefix[lid.x], flag == 1u);
        if (lid.x == BLOCK_SIZE - 1u) {
            block_sums[tile] = value;
        }
    }
}
