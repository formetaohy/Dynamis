@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> contacts: array<Contact>;
@group(0) @binding(2) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(3) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> segments: array<u32>;
@group(0) @binding(5) var<storage, read_write> block_count: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    if (lid.x != 0u) {
        return;
    }
    let contact_blocks = min(atomicLoad(&contact_count[0]), arrayLength(&contacts));
    let constraint_blocks = min(params.constraint_count, arrayLength(&constraint_runtime));
    segments[SOLVER_BLOCK_CONTACT] = contact_blocks;
    segments[SOLVER_BLOCK_CONSTRAINT] = constraint_blocks;
    atomicStore(&block_count[0], contact_blocks + constraint_blocks);
}
