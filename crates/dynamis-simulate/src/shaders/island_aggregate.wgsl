@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read_write> island_parents: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> island_state: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> wake_flags: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= params.body_count) {
        return;
    }
    let body = bodies[index];
    let root = atomicLoad(&island_parents[index]);
    if (atomicLoad(&wake_flags[index]) != 0u) {
        atomicMax(&island_state[root], ISLAND_WAKE);
    }
    if (body_is_dynamic(body) && (body.flags & BODY_SLEEPING) == 0u) {
        let speed = length(body.velocity);
        let spin = length(body.angular_velocity);
        let sleep_velocity = select(params.sleep_velocity, body.sleep_velocity_override, body.sleep_velocity_override > 0.0);
        let sleep_angular_velocity = select(params.sleep_angular_velocity, body.sleep_angular_velocity_override, body.sleep_angular_velocity_override > 0.0);
        if (speed > sleep_velocity || spin > sleep_angular_velocity) {
            atomicMax(&island_state[root], ISLAND_ACTIVE);
        }
    }
}
