@group(0) @binding(0) var<storage, read> segments: array<u32>;
@group(0) @binding(1) var<storage, read> a_bodies: array<u32>;
@group(0) @binding(2) var<storage, read> b_bodies: array<u32>;
@group(0) @binding(3) var<storage, read_write> first_a: array<u32>;
@group(0) @binding(4) var<storage, read_write> first_b: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let slot = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (slot >= segments[SOLVER_BLOCK_CONTACT] + segments[SOLVER_BLOCK_CONSTRAINT]) {
        return;
    }
    let key_a = a_bodies[slot];
    if (slot == 0u || a_bodies[slot - 1u] != key_a) {
        first_a[key_a] = slot + 1u;
    }
    let key_b = b_bodies[slot];
    if (slot == 0u || b_bodies[slot - 1u] != key_b) {
        first_b[key_b] = slot + 1u;
    }
}
