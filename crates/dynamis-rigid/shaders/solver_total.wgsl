@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> contacts: array<Contact>;
@group(0) @binding(2) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> block_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> solver_row_count: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    if (lid.x != 0u) {
        return;
    }
    atomicStore(&block_count[0], min(atomicLoad(&contact_count[0]), arrayLength(&contacts)));
    atomicStore(&solver_row_count[0], 0u);
}
