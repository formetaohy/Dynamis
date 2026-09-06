const RADIX_BITS: u32 = 8u;
const BIN_COUNT: u32 = 256u;
const HISTOGRAM_THREADS: u32 = 256u;
const SHIFT: u32 = __SHIFT__u;

@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, BIN_COUNT>;
@group(0) @binding(3) var<storage, read> count_holder: array<u32>;

@compute @workgroup_size(HISTOGRAM_THREADS)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let count = count_holder[0];
    var index = gid.x;
    while (index < count) {
        let key = select(keys_hi[index], keys_lo[index], SHIFT < 32u);
        let digit = (key >> (SHIFT & 31u)) & 0xFFu;
        atomicAdd(&histogram[digit], 1u);
        index = index + HISTOGRAM_THREADS;
    }
}
