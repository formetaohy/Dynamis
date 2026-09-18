@group(0) @binding(0) var<storage, read> counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_totals: array<u32>;
@group(0) @binding(3) var<storage, read> length_holder: array<u32>;
@group(0) @binding(4) var<storage, read> base_holder: array<u32>;

__PARTITION__

@compute @workgroup_size(TILE)
fn main(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(num_workgroups) groups: vec3u,
) {
    let bin = lid.x;
    let chunk = wgid.y * UNITS_PER_ROW + wgid.x;
    let units = min(sort_units(sort_length()), groups.x * groups.y * CHUNK);
    let first = chunk * CHUNK;
    let last = min(first + CHUNK, units);
    var total = 0u;
    for (var unit = first; unit < last; unit = unit + 1u) {
        let at = unit * BINS + bin;
        offsets[at] = total;
        total = total + counts[at];
    }
    block_totals[chunk * BINS + bin] = total;
}
