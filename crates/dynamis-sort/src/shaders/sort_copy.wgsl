@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read> payload_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_lo_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_hi_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> payload_out: array<u32>;
@group(0) @binding(6) var<storage, read> length_holder: array<u32>;
@group(0) @binding(7) var<storage, read> base_holder: array<u32>;

__PARTITION__

@compute @workgroup_size(TILE)
fn main(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(num_workgroups) groups: vec3u,
) {
    let unit = wgid.y * UNITS_PER_ROW + wgid.x;
    let length = sort_length();
    let origin = sort_base();
    let stride = groups.x * groups.y * TILE;
    for (
        var index = unit * TILE + lid.x;
        index < length;
        index = index + stride
    ) {
        keys_lo_out[origin + index] = keys_lo[origin + index];
        keys_hi_out[origin + index] = keys_hi[origin + index];
        payload_out[origin + index] = payload_in[origin + index];
    }
}
