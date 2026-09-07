@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, 256>;
@group(0) @binding(3) var<storage, read_write> block_histogram: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> count_holder: array<u32>;

const BIN_COUNT: u32 = 256u;
const TILE_SIZE: u32 = 256u;
const SHIFT: u32 = __SHIFT__u;

@compute @workgroup_size(TILE_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= count_holder[0]) {
        return;
    }
    let key = select(keys_hi[index], keys_lo[index], SHIFT < 32u);
    let digit = (key >> (SHIFT & 31u)) & 0xFFu;
    atomicAdd(&histogram[digit], 1u);
    atomicAdd(&block_histogram[(index / TILE_SIZE) * BIN_COUNT + digit], 1u);
}
