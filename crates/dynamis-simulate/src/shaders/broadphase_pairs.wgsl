@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> entry_keys_hi: array<u32>;
@group(0) @binding(2) var<storage, read> entry_keys_lo: array<u32>;
@group(0) @binding(3) var<storage, read_write> entry_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> pair_keys_hi: array<u32>;
@group(0) @binding(5) var<storage, read_write> pair_keys_lo: array<u32>;
@group(0) @binding(6) var<storage, read_write> pair_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> spillover: array<atomic<u32>>;

fn emit_pair(first: u32, second: u32) {
    if (first == second) {
        return;
    }
    let a = min(first, second);
    let b = max(first, second);
    let slot = atomicAdd(&pair_count[0], 1u);
    if (slot < arrayLength(&pair_keys_lo)) {
        pair_keys_lo[slot] = b;
        pair_keys_hi[slot] = a;
    } else {
        atomicAdd(&spillover[0], 1u);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    let live = min(atomicLoad(&entry_count[0]), arrayLength(&entry_keys_hi));
    if (index >= live) {
        return;
    }
    let cell = entry_keys_hi[index];
    var cursor = index + 1u;
    while (cursor < live && entry_keys_hi[cursor] == cell) {
        emit_pair(entry_keys_lo[index], entry_keys_lo[cursor]);
        cursor = cursor + 1u;
    }
}
