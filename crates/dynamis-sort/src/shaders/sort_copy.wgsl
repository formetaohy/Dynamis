@group(0) @binding(0) var<storage, read> keys_lo_in: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi_in: array<u32>;
@group(0) @binding(2) var<storage, read> payload_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_lo_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_hi_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> payload_out: array<u32>;
@group(0) @binding(6) var<storage, read> count_holder: array<u32>;

const ROW: u32 = __ROW__u;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let length = min(count_holder[0], arrayLength(&keys_lo_in));
    let stride = groups.x * groups.y * 256u;
    for (
        var index = (wgid.y * ROW + wgid.x) * 256u + lid.x;
        index < length;
        index = index + stride
    ) {
        keys_lo_out[index] = keys_lo_in[index];
        keys_hi_out[index] = keys_hi_in[index];
        payload_out[index] = payload_in[index];
    }
}
