@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read_write> cursor: array<atomic<u32>, __BUCKETS__>;
@group(0) @binding(3) var<storage, read> block_prefix: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> values_out: array<u32>;
@group(0) @binding(6) var<storage, read> count_holder: array<u32>;

const BUCKET_COUNT: u32 = __BUCKETS__u;
const BLOCK_SIZE: u32 = 256u;

var<workgroup> digit_map: array<u32, BLOCK_SIZE>;
var<workgroup> local_rank: array<u32, BLOCK_SIZE>;
var<workgroup> bin_count: array<atomic<u32>, BUCKET_COUNT>;

@compute @workgroup_size(BLOCK_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let index = gid.x;
    let lane = lid.x;
    if (lane < BUCKET_COUNT) {
        bin_count[lane] = 0u;
    }
    workgroupBarrier();
    let valid = index < count_holder[0];
    var digit = 0u;
    if (valid) {
        digit = keys[index];
        digit_map[lane] = digit;
        atomicAdd(&bin_count[digit], 1u);
    } else {
        digit_map[lane] = BUCKET_COUNT;
    }
    workgroupBarrier();
    let scan_digit = lid.x;
    var rank = 0u;
    for (var lane_i = 0u; lane_i < BLOCK_SIZE; lane_i = lane_i + 1u) {
        if (digit_map[lane_i] == scan_digit) {
            local_rank[lane_i] = rank;
            rank = rank + 1u;
        }
    }
    workgroupBarrier();
    if (valid && lid.x < BLOCK_SIZE) {
        let block = index / BLOCK_SIZE;
        let out_index = atomicLoad(&cursor[digit]) + block_prefix[block * BUCKET_COUNT + digit] + local_rank[lane];
        keys_out[out_index] = keys[index];
        values_out[out_index] = values[index];
    }
}
