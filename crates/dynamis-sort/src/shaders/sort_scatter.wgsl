@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read> payload_in: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> block_totals: array<u32>;
@group(0) @binding(5) var<storage, read_write> keys_lo_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> keys_hi_out: array<u32>;
@group(0) @binding(7) var<storage, read_write> payload_out: array<u32>;
@group(0) @binding(8) var<storage, read> length_holder: array<u32>;

__PARTITION__

const EMPTY: u32 = 0xFFFFFFFFu;

var<workgroup> running: array<u32, BINS>;
var<workgroup> totals: array<u32, BINS>;
var<workgroup> digit_base: array<u32, BINS>;
var<workgroup> digit_map: array<u32, TILE>;
var<workgroup> local_rank: array<u32, TILE>;

@compute @workgroup_size(TILE)
fn main(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let lane = lid.x;
    let unit = wgid.y * UNITS_PER_ROW + wgid.x;
    let length = sort_length();
    if (unit >= sort_units(length)) {
        return;
    }
    let chunk = unit / CHUNK;
    var total = 0u;
    var prefix = 0u;
    for (var other = 0u; other < CHUNKS; other = other + 1u) {
        let count = block_totals[other * BINS + lane];
        total = total + count;
        if (other < chunk) {
            prefix = prefix + count;
        }
    }
    totals[lane] = total;
    running[lane] = 0u;
    workgroupBarrier();
    var base = prefix;
    for (var bin = 0u; bin < lane; bin = bin + 1u) {
        base = base + totals[bin];
    }
    digit_base[lane] = base;
    workgroupBarrier();
    let tiles = sort_tiles(unit, length);
    let at = unit * BINS;
    for (var tile = tiles.first; tile < tiles.last; tile = tile + 1u) {
        let index = tile * TILE + lane;
        let valid = index < length;
        var digit = EMPTY;
        if (valid) {
            let key = select(keys_lo[index], keys_hi[index], DIGIT_WORD != 0u);
            digit = (key >> DIGIT_SHIFT) & BINS_MASK;
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
            let place = digit_base[digit] + offsets[at + digit] + running[digit] + local_rank[lane];
            keys_lo_out[place] = keys_lo[index];
            keys_hi_out[place] = keys_hi[index];
            payload_out[place] = payload_in[index];
        }
        workgroupBarrier();
        running[lane] = running[lane] + rank;
    }
}
