@group(0) @binding(0) var<storage, read_write> class_counts: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> class_bounds: array<u32>;
@group(0) @binding(2) var<storage, read_write> class_cursors: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> overflow_count: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    if (lid.x != 0u) {
        return;
    }
    var total = 0u;
    for (var bucket = 0u; bucket < SOLVER_CLASS_BUCKETS; bucket = bucket + 1u) {
        let count = atomicLoad(&class_counts[bucket]);
        atomicStore(&class_counts[bucket], 0u);
        let row = bucket * SOLVER_CLASS_ROW_WORDS;
        class_bounds[row] = total;
        class_bounds[row + 1u] = total + count;
        atomicStore(&class_cursors[bucket], total);
        total = total + count;
    }
    overflow_count[0] =
        total - class_bounds[SOLVER_CLASS_OVERFLOW * SOLVER_CLASS_ROW_WORDS];
}
