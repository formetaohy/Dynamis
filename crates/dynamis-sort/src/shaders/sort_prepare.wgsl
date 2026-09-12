@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, 2048>;
@group(0) @binding(3) var<storage, read_write> histogram_free: array<atomic<u32>, 2048>;
@group(0) @binding(4) var<storage, read_write> rows: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> count_holder: array<u32>;

const BINS: u32 = 256u;
const BINS_ALL: u32 = 2048u;
const REGIONS: u32 = 8u;
const SLOTS: u32 = __SLOTS__u;
const ROW: u32 = __ROW__u;
const REDUCTION: u32 = 1u;

var<workgroup> bins: array<atomic<u32>, BINS_ALL>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let lane = lid.x;
    let unit = wgid.y * ROW + wgid.x;
    for (var region = 0u; region < REGIONS; region = region + 1u) {
        atomicStore(&rows[(region * SLOTS + unit) * BINS + lane], REDUCTION);
    }
    for (var digit = 0u; digit < 8u; digit = digit + 1u) {
        atomicStore(&bins[digit * BINS + lane], 0u);
    }
    workgroupBarrier();
    let length = min(count_holder[0], arrayLength(&keys_lo));
    let spans = (length + 255u) / 256u;
    var span = (spans + groups.x * groups.y - 1u) / (groups.x * groups.y);
    if (span == 0u) {
        span = 1u;
    }
    let first = min(unit * span, spans);
    let last = min(first + span, spans);
    for (var tile = first; tile < last; tile = tile + 1u) {
        let index = tile * 256u + lane;
        if (index < length) {
            for (var digit = 0u; digit < 8u; digit = digit + 1u) {
                let word = select(keys_lo[index], keys_hi[index], digit >= 4u);
                atomicAdd(&bins[digit * BINS + ((word >> ((digit & 3u) * 8u)) & 255u)], 1u);
            }
        }
    }
    workgroupBarrier();
    for (var digit = 0u; digit < 8u; digit = digit + 1u) {
        atomicStore(
            &rows[(digit * SLOTS + unit) * BINS + lane],
            (atomicLoad(&bins[digit * BINS + lane]) << 2u) | REDUCTION,
        );
        atomicAdd(
            &histogram[digit * BINS + lane],
            atomicLoad(&bins[digit * BINS + lane]),
        );
    }
    if (unit == 0u) {
        for (var bin = lane; bin < BINS_ALL; bin = bin + 256u) {
            atomicStore(&histogram_free[bin], 0u);
        }
    }
}
