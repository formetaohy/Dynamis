@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read_write> rows: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> count_holder: array<u32>;

const BINS: u32 = 256u;
const TILE: u32 = 256u;
const SHIFT: u32 = __SHIFT__u;
const REGION: u32 = __REGION__u;
const REDUCTION: u32 = 1u;
const ROW: u32 = __ROW__u;

var<workgroup> bins: array<atomic<u32>, BINS>;

@compute @workgroup_size(TILE)
fn main(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let lane = lid.x;
    let unit = wgid.y * ROW + wgid.x;
    let units = groups.x * groups.y;
    let length = min(count_holder[0], arrayLength(&keys_lo));
    let spans = (length + TILE - 1u) / TILE;
    var span = (spans + units - 1u) / units;
    if (span == 0u) {
        span = 1u;
    }
    let first = min(unit * span, spans);
    let last = min(first + span, spans);
    atomicStore(&bins[lane], 0u);
    workgroupBarrier();
    for (var tile = first; tile < last; tile = tile + 1u) {
        let index = tile * TILE + lane;
        if (index < length) {
            let key = select(keys_hi[index], keys_lo[index], SHIFT < 32u);
            atomicAdd(&bins[(key >> (SHIFT & 31u)) & 0xFFu], 1u);
        }
    }
    workgroupBarrier();
    atomicStore(
        &rows[REGION + unit * BINS + lane],
        (atomicLoad(&bins[lane]) << 2u) | REDUCTION,
    );
}
