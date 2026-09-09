@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read_write> island_parents: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> island_state: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> wake_flags: array<atomic<u32>>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (index >= params.dynamic_count) {
        return;
    }
    let body = load_body(index);
    let root = atomicLoad(&island_parents[index]);
    if (atomicLoad(&wake_flags[index]) != 0u) {
        atomicMax(&island_state[root], ISLAND_WAKE);
    }
    if (body_is_dynamic(body) && body.state.sleeping == 0u) {
        let speed = length(body.state.velocity);
        let spin = length(body.state.angular_velocity);
        let sleep_velocity = select(params.sleep_velocity, body.desc.sleep_velocity, (body.desc.flags & OVERRIDE_SLEEP_LINEAR) != 0u);
        let sleep_angular_velocity = select(params.sleep_angular_velocity, body.desc.sleep_angular_velocity, (body.desc.flags & OVERRIDE_SLEEP_ANGULAR) != 0u);
        if (speed > sleep_velocity || spin > sleep_angular_velocity) {
            atomicMax(&island_state[root], ISLAND_ACTIVE);
        }
    }
}
