@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;
@group(0) @binding(3) var<storage, read> length_holder: array<u32>;

__PARTITION__

var<workgroup> bins: array<atomic<u32>, BINS>;

@compute @workgroup_size(TILE)
fn main(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let lane = lid.x;
    let unit = wgid.y * UNITS_PER_ROW + wgid.x;
    atomicStore(&bins[lane], 0u);
    workgroupBarrier();
    let length = sort_length();
    if (unit < sort_units(length)) {
        let tiles = sort_tiles(unit, length);
        for (var tile = tiles.first; tile < tiles.last; tile = tile + 1u) {
            let index = tile * TILE + lane;
            if (index < length) {
                let key = select(keys_lo[index], keys_hi[index], DIGIT_WORD != 0u);
                atomicAdd(&bins[(key >> DIGIT_SHIFT) & BINS_MASK], 1u);
            }
        }
    }
    workgroupBarrier();
    counts[unit * BINS + lane] = atomicLoad(&bins[lane]);
}
