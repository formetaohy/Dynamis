@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>, __BUCKETS__>;
@group(0) @binding(1) var<storage, read_write> cursor: array<atomic<u32>, __BUCKETS__>;

const BUCKET_COUNT: u32 = __BUCKETS__u;

var<workgroup> scratch: array<u32, 256>;

@compute @workgroup_size(256u)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let index = lid.x;
    let per = (BUCKET_COUNT + 255u) / 256u;
    var local = 0u;
    for (var offset = 0u; offset < per; offset = offset + 1u) {
        let bucket = index * per + offset;
        if (bucket >= BUCKET_COUNT) {
            break;
        }
        local = local + atomicLoad(&histogram[bucket]);
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
        let bucket = index * per + offset;
        if (bucket >= BUCKET_COUNT) {
            break;
        }
        let count = atomicLoad(&histogram[bucket]);
        atomicStore(&cursor[bucket], group_base);
        atomicStore(&histogram[bucket], 0u);
        group_base = group_base + count;
    }
}
