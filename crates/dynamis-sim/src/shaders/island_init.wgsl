@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> island_parents: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> island_state: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.body_count) {
        return;
    }
    atomicStore(&island_parents[index], index);
    atomicStore(&island_state[index], 0u);
}
