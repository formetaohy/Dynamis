@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> first_a: array<u32>;
@group(0) @binding(2) var<storage, read_write> first_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> contact_counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> resolution: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> contributions: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.body_count) {
        return;
    }
    first_a[index] = 0u;
    first_b[index] = 0u;
    block_counts[index] = 0u;
    contact_counts[index] = 0u;
    resolution[index] = vec4f(0.0);
    atomicStore(&contributions[index], 0u);
}
