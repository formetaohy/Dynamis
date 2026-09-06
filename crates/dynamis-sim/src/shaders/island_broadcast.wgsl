@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read> island_parents: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> island_state: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> wake_flags: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.body_count) {
        return;
    }
    var body = bodies[index];
    let root = atomicLoad(&island_parents[index]);
    let state = atomicLoad(&island_state[root]);
    if ((state & ISLAND_WAKE) != 0u) {
        body.flags = body.flags & ~BODY_SLEEPING;
        body.sleep_timer = 0.0;
    } else if (body_is_dynamic(body) && (body.flags & BODY_SLEEPING) == 0u) {
        if ((state & ISLAND_ACTIVE) != 0u) {
            body.sleep_timer = 0.0;
        } else {
            body.sleep_timer = body.sleep_timer + params.dt;
            if (body.sleep_timer >= params.sleep_time) {
                body.flags = body.flags | BODY_SLEEPING;
                body.velocity = vec3f(0.0);
                body.angular_velocity = vec3f(0.0);
            }
        }
    }
    atomicStore(&wake_flags[index], 0u);
    bodies[index] = body;
}
