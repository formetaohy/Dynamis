@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(2) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(3) var<storage, read_write> joint_major: array<u32>;
@group(0) @binding(4) var<storage, read_write> joint_minor: array<u32>;
@group(0) @binding(5) var<storage, read_write> joint_count: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> constraint_rows: array<ConstraintRows>;

fn work(index: u32) {
    if constraint_runtime[index].broken != 0u {
        return;
    }
    let constraint = constraint_descs[index];
    if ((constraint.flags & CONSTRAINT_DISABLE_COLLISIONS) == 0u) {
        return;
    }
    let rows = constraint_rows[index];
    let a = min(rows.first_row, rows.second_row);
    let b = max(rows.first_row, rows.second_row);
    let slot = atomicAdd(&joint_count[0], 1u);
    if (slot < arrayLength(&joint_major)) {
        joint_major[slot] = a;
        joint_minor[slot] = b;
    }
}
