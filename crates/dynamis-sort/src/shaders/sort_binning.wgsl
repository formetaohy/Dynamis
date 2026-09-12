@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read> payload_in: array<u32>;
@group(0) @binding(3) var<storage, read> histogram: array<u32>;
@group(0) @binding(4) var<storage, read_write> rows: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> keys_lo_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> keys_hi_out: array<u32>;
@group(0) @binding(7) var<storage, read_write> payload_out: array<u32>;
@group(0) @binding(8) var<storage, read> count_holder: array<u32>;

const BINS: u32 = 256u;
const TILE: u32 = 256u;
const SHIFT: u32 = __SHIFT__u;
const REGION: u32 = __REGION__u;
const DIGIT: u32 = __DIGIT__u;
const INCLUSIVE: u32 = 2u;
const STATE: u32 = 3u;
const EMPTY: u32 = 0xFFFFFFFFu;
const ROW: u32 = __ROW__u;

var<workgroup> running: array<u32, BINS>;
var<workgroup> bases: array<u32, BINS>;
var<workgroup> offsets: array<u32, BINS>;
var<workgroup> digit_map: array<u32, TILE>;
var<workgroup> local_rank: array<u32, TILE>;

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

    if (first >= spans) {
        return;
    }
    let own = atomicLoad(&rows[REGION + unit * BINS + lane]) >> 2u;
    var total = own;
    var back = unit;
    loop {
        if (back == 0u) {
            break;
        }
        back = back - 1u;
        let entry = atomicLoad(&rows[REGION + back * BINS + lane]);
        total = total + (entry >> 2u);
        if ((entry & STATE) == INCLUSIVE) {
            break;
        }
    }
    atomicStore(&rows[REGION + unit * BINS + lane], (total << 2u) | INCLUSIVE);

    bases[lane] = histogram[DIGIT * BINS + lane];
    workgroupBarrier();
    var base = 0u;
    for (var bin = 0u; bin < lane; bin = bin + 1u) {
        base = base + bases[bin];
    }
    offsets[lane] = base + total - own;
    running[lane] = 0u;
    workgroupBarrier();

    for (var tile = first; tile < last; tile = tile + 1u) {
        let index = tile * TILE + lane;
        let valid = index < length;
        var digit = EMPTY;
        if (valid) {
            let key = select(keys_hi[index], keys_lo[index], SHIFT < 32u);
            digit = (key >> (SHIFT & 31u)) & 0xFFu;
        }
        digit_map[lane] = digit;
        workgroupBarrier();
        var rank = 0u;
        for (var peer = 0u; peer < TILE; peer = peer + 1u) {
            if (digit_map[peer] == lane) {
                local_rank[peer] = rank;
                rank = rank + 1u;
            }
        }
        workgroupBarrier();
        if (valid) {
            let place = offsets[digit] + running[digit] + local_rank[lane];
            keys_lo_out[place] = keys_lo[index];
            keys_hi_out[place] = keys_hi[index];
            payload_out[place] = payload_in[index];
        }
        workgroupBarrier();
        running[lane] = running[lane] + rank;
    }
}
