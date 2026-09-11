@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, 256>;
@group(0) @binding(3) var<storage, read_write> block_histogram: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> count_holder: array<u32>;

const BIN_COUNT: u32 = 256u;
const TILE_SIZE: u32 = 256u;
const SHIFT: u32 = __SHIFT__u;

@compute @workgroup_size(TILE_SIZE)
fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let length = min(count_holder[0], arrayLength(&keys_lo));
    let tiles = groups.x * groups.y;
    for (var tile = wgid.y * __ROW__ + wgid.x; tile * TILE_SIZE < length; tile = tile + tiles) {
        let index = tile * TILE_SIZE + lid.x;
        if (index >= length) {
            continue;
        }
        let key = select(keys_hi[index], keys_lo[index], SHIFT < 32u);
        let digit = (key >> (SHIFT & 31u)) & 0xFFu;
        atomicAdd(&histogram[digit], 1u);
        atomicAdd(&block_histogram[tile * BIN_COUNT + digit], 1u);
    }
}
