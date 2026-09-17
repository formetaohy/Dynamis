@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read> payload_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_lo_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_hi_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> payload_out: array<u32>;
@group(0) @binding(6) var<storage, read> length_holder: array<u32>;

__PARTITION__

@compute @workgroup_size(TILE)
fn main(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let unit = wgid.y * UNITS_PER_ROW + wgid.x;
    let length = sort_length();
    let stride = sort_units(length) * TILE;
    for (
        var index = unit * TILE + lid.x;
        index < length;
        index = index + stride
    ) {
        keys_lo_out[index] = keys_lo[index];
        keys_hi_out[index] = keys_hi[index];
        payload_out[index] = payload_in[index];
    }
}
