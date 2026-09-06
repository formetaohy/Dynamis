const RADIX_BITS: u32 = 8u;
const SCATTER_THREADS: u32 = 256u;
const SHIFT: u32 = __SHIFT__u;

@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<u32>;
@group(0) @binding(3) var<storage, read_write> buckets: array<atomic<u32>, 256u>;
@group(0) @binding(4) var<storage, read_write> keys_lo_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> keys_hi_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> values_out: array<u32>;
@group(0) @binding(7) var<storage, read> count_holder: array<u32>;

@compute @workgroup_size(SCATTER_THREADS)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let count = count_holder[0];
    var index = count - 1u - gid.x;
    while (index < count) {
        let key = select(keys_hi[index], keys_lo[index], SHIFT < 32u);
        let digit = (key >> (SHIFT & 31u)) & 0xFFu;
        let slot = atomicSub(&buckets[digit], 1u) - 1u;
        if (slot < arrayLength(&keys_lo_out)) {
            keys_lo_out[slot] = keys_lo[index];
            keys_hi_out[slot] = keys_hi[index];
            values_out[slot] = values[index];
        }
        index = index - SCATTER_THREADS;
    }
}
