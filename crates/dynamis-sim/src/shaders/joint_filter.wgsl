@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> constraints: array<Constraint>;
@group(0) @binding(2) var<storage, read_write> joint_hi: array<u32>;
@group(0) @binding(3) var<storage, read_write> joint_lo: array<u32>;
@group(0) @binding(4) var<storage, read_write> joint_count: atomic<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.constraint_count) {
        return;
    }
    let constraint = constraints[index];
    if (constraint.kind == CONSTRAINT_INVALID) {
        return;
    }
    if ((constraint.flags & CONSTRAINT_DISABLE_COLLISIONS) == 0u) {
        return;
    }
    let a = min(constraint.a, constraint.b);
    let b = max(constraint.a, constraint.b);
    let slot = atomicAdd(&joint_count, 1u);
    if (slot < arrayLength(&joint_hi)) {
        joint_hi[slot] = a;
        joint_lo[slot] = b;
    }
}
