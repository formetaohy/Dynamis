@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> island_parents: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.dynamic_count) {
        return;
    }
    let parent = atomicLoad(&island_parents[index]);
    let grandparent = atomicLoad(&island_parents[parent]);
    island_parents[index] = min(parent, grandparent);
}
