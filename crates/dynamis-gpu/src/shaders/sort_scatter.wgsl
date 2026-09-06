@group(0) @binding(0) var<storage, read> keys_lo: array<u32>;
@group(0) @binding(1) var<storage, read> keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<u32>;
@group(0) @binding(3) var<storage, read> positions: array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_lo_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> keys_hi_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> values_out: array<u32>;
@group(0) @binding(7) var<storage, read> count_holder: array<u32>;

@compute @workgroup_size(256u)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= count_holder[0]) {
        return;
    }
    let slot = positions[index];
    if (slot < arrayLength(&keys_lo_out)) {
        keys_lo_out[slot] = keys_lo[index];
        keys_hi_out[slot] = keys_hi[index];
        values_out[slot] = values[index];
    }
}
