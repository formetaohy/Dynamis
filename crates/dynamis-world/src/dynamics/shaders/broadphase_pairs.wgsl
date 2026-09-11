@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> entry_cells: array<u32>;
@group(0) @binding(2) var<storage, read> entry_colliders: array<u32>;
@group(0) @binding(3) var<storage, read_write> entry_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> pair_major: array<u32>;
@group(0) @binding(5) var<storage, read_write> pair_minor: array<u32>;
@group(0) @binding(6) var<storage, read_write> pair_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> spillover: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> body_activity: array<u32>;

fn collider_is_awake(collider: u32) -> bool {
    return body_activity[collider / MAX_COLLIDERS_PER_BODY] != 0u;
}

fn emit_pair(first: u32, second: u32) {
    if (first == second) {
        return;
    }
    let a = min(first, second);
    let b = max(first, second);
    let slot = atomicAdd(&pair_count[0], 1u);
    if (slot < arrayLength(&pair_minor)) {
        pair_minor[slot] = b;
        pair_major[slot] = a;
    } else {
        atomicAdd(&spillover[0], 1u);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = min(atomicLoad(&entry_count[0]), arrayLength(&entry_cells));
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {
        if (!collider_is_awake(entry_colliders[index])) {
            continue;
        }
        let cell = entry_cells[index];
        var cursor = index + 1u;
        while (cursor < live && entry_cells[cursor] == cell) {
            emit_pair(entry_colliders[index], entry_colliders[cursor]);
            cursor = cursor + 1u;
        }
        var back = index;
        while (back > 0u && entry_cells[back - 1u] == cell) {
            back = back - 1u;
            if (!collider_is_awake(entry_colliders[back])) {
                emit_pair(entry_colliders[back], entry_colliders[index]);
            }
        }
    }
}
