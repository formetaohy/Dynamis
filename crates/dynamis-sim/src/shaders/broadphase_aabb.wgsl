@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read_write> aabbs: array<Aabb>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.body_count) {
        return;
    }
    let body = bodies[index];
    let radius = vec3f(body.radius);
    aabbs[index] = Aabb(body.position - radius, 0.0, body.position + radius, 0.0);
}
