const BIN_COUNT: u32 = 256u;
const TILE_SIZE: u32 = 256u;
const SHIFT: u32 = __SHIFT__u;

@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<u32>;
@group(0) @binding(3) var<storage, read> cursor: array<u32, BIN_COUNT>;
@group(0) @binding(4) var<storage, read> block_prefix: array<u32>;
@group(0) @binding(5) var<storage, read_write> keys_lo_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> keys_hi_out: array<u32>;
@group(0) @binding(7) var<storage, read_write> values_out: array<u32>;
@group(0) @binding(8) var<storage, read> count_holder: array<u32>;

var<workgroup> digit_map: array<u32, TILE_SIZE>;
var<workgroup> local_rank: array<u32, TILE_SIZE>;
var<workgroup> bin_count: array<atomic<u32>, BIN_COUNT>;

@compute @workgroup_size(TILE_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let index = gid.x;
    let lane = lid.x;
    bin_count[lane] = 0u;
    workgroupBarrier();
    let valid = index < min(count_holder[0], arrayLength(&keys_lo));
    var digit = 0u;
    if (valid) {
        let key = select(keys_hi[index], keys_lo[index], SHIFT < 32u);
        digit = (key >> (SHIFT & 31u)) & 0xFFu;
        digit_map[lane] = digit;
        atomicAdd(&bin_count[digit], 1u);
    } else {
        digit_map[lane] = 0xFFFFFFFFu;
    }
    workgroupBarrier();
    let scan_digit = lid.x;
    var rank = 0u;
    for (var lane_i = 0u; lane_i < TILE_SIZE; lane_i = lane_i + 1u) {
        if (digit_map[lane_i] == scan_digit) {
            local_rank[lane_i] = rank;
            rank = rank + 1u;
        }
    }
    workgroupBarrier();
    if (valid) {
        let block = index / TILE_SIZE;
        let out_index = cursor[digit] + block_prefix[block * BIN_COUNT + digit] + local_rank[lane];
        keys_lo_out[out_index] = keys_lo[index];
        keys_hi_out[out_index] = keys_hi[index];
        values_out[out_index] = values[index];
    }
}
