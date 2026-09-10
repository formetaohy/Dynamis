@group(0) @binding(0) var<storage, read_write> slept: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> gathered: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> index_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> deferred_woke: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> pending: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    if (lid.x != 0u) {
        return;
    }
    if (atomicLoad(&slept[0]) > 0u) {
        atomicStore(&index_count[0], atomicLoad(&gathered[0]));
    }
    atomicStore(&pending[0], select(0u, 1u, atomicLoad(&deferred_woke[0]) > 0u));
}
