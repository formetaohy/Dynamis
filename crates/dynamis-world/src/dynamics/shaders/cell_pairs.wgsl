@group(0) @binding(3) var<uniform> params: StepParams;
@group(0) @binding(4) var<storage, read_write> pair_major: array<u32>;
@group(0) @binding(5) var<storage, read_write> pair_minor: array<u32>;
@group(0) @binding(6) var<storage, read> body_activity: array<u32>;
@group(0) @binding(7) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(8) var<storage, read> aabbs: array<Aabb>;

fn collider_is_awake(collider: u32) -> bool {
    let owner = collider_owners[collider];
    return owner != NO_BODY && body_activity[owner] != 0u;
}

fn emit_pair(first: u32, second: u32) {
    let owner = collider_owners[first];
    if (first == second || owner == collider_owners[second]) {
        return;
    }
    let slot = counter_add(COUNTER_PAIRS, 1u);
    if (slot < arrayLength(&pair_minor)) {
        let a = min(first, second);
        let b = max(first, second);
        pair_minor[slot] = b;
        pair_major[slot] = a;
    } else {
        counter_add(COUNTER_SPILLOVER_PAIRS, 1u);
    }
}

fn emit_cell_mate(first: u32, second: u32) {
    if (!aabb_overlaps(aabbs[first], aabbs[second])) {
        return;
    }
    emit_pair(first, second);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let live = entry_live();
    let stride = grid_stride(groups);
    for (var index = global_index(gid); index < live; index = index + stride) {
        if (!collider_is_awake(entry_colliders[index])) {
            continue;
        }
        let key = entry_keys[index];
        var cursor = index + 1u;
        while (cursor < live && entry_keys[cursor] == key) {
            emit_cell_mate(entry_colliders[index], entry_colliders[cursor]);
            cursor = cursor + 1u;
        }
        var back = index;
        while (back > 0u && entry_keys[back - 1u] == key) {
            back = back - 1u;
            if (!collider_is_awake(entry_colliders[back])) {
                emit_cell_mate(entry_colliders[back], entry_colliders[index]);
            }
        }
    }
}
