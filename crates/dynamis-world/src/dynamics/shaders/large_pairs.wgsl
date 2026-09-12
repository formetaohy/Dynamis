@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> large_bodies: array<u32>;
@group(0) @binding(2) var<storage, read_write> large_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> pair_major: array<u32>;
@group(0) @binding(4) var<storage, read_write> pair_minor: array<u32>;
@group(0) @binding(5) var<storage, read_write> pair_count: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> colliders: array<Collider>;
@group(0) @binding(7) var<storage, read_write> spillover: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> body_activity: array<u32>;
@group(0) @binding(9) var<storage, read> collider_owners: array<u32>;

fn emit_pair(first: u32, second: u32) {
    let owner = collider_owners[first];
    if (first == second || owner == collider_owners[second]) {
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
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.collider_count) {
        return;
    }
    let live = min(atomicLoad(&large_count[0]), arrayLength(&large_bodies));
    if (live == 0u) {
        return;
    }
    let collider = colliders[index];
    let owner = collider_owners[index];
    if (collider.kind == SHAPE_NONE || owner == NO_BODY) {
        return;
    }
    let origin_active = body_activity[owner] != 0u;
    for (var large = 0u; large < live; large = large + 1u) {
        let large_index = large_bodies[large];
        let large_owner = collider_owners[large_index];
        if (origin_active || (large_owner != NO_BODY && body_activity[large_owner] != 0u)) {
            emit_pair(large_index, index);
        }
    }
}
