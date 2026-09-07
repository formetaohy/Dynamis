@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read> constraints: array<Constraint>;
@group(0) @binding(3) var<storage, read_write> island_parents: array<atomic<u32>>;

fn island_link(a: u32, b: u32) {
    atomicMin(&island_parents[a], b);
    atomicMin(&island_parents[b], a);
}

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
    let first = bodies[constraint.a];
    let second = bodies[constraint.b];
    if (body_is_dynamic(first) && body_is_dynamic(second)) {
        island_link(constraint.a, constraint.b);
    }
}
