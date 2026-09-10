@group(0) @binding(0) var<storage, read> a_body: array<u32>;
@group(0) @binding(1) var<storage, read> b_bodies: array<u32>;
@group(0) @binding(2) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> first_a: array<u32>;
@group(0) @binding(4) var<storage, read_write> first_b: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let value = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (value >= min(atomicLoad(&contact_count[0]), arrayLength(&a_body))) {
        return;
    }
    let key_a = a_body[value];
    if (value == 0u || a_body[value - 1u] != key_a) {
        first_a[key_a] = value + 1u;
    }
    let key_b = b_bodies[value];
    if (value == 0u || b_bodies[value - 1u] != key_b) {
        first_b[key_b] = value + 1u;
    }
}
