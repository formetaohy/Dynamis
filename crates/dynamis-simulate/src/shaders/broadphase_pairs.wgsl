@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> entry_keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read> entry_keys_lo: array<u32>;
@group(0) @binding(3) var<storage, read_write> entry_count: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> pair_keys_hi: array<u32>;
@group(0) @binding(5) var<storage, read_write> pair_keys_lo: array<u32>;
@group(0) @binding(6) var<storage, read_write> pair_count: atomic<u32>;
@group(0) @binding(7) var<storage, read_write> overflow: array<atomic<u32>>;

fn emit_pair(first: u32, second: u32) {
    if (first == second) {
        return;
    }
    let a = min(first, second);
    let b = max(first, second);
    let slot = atomicAdd(&pair_count, 1u);
    if (slot < arrayLength(&pair_keys_lo)) {
        pair_keys_lo[slot] = b;
        pair_keys_hi[slot] = a;
    } else {
        atomicAdd(&overflow[0u], 1u);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= atomicLoad(&entry_count)) {
        return;
    }
    let cell = entry_keys_hi[index];
    var cursor = index + 1u;
    while (cursor < atomicLoad(&entry_count) && entry_keys_hi[cursor] == cell) {
        emit_pair(entry_keys_lo[index], entry_keys_lo[cursor]);
        cursor = cursor + 1u;
    }
}
