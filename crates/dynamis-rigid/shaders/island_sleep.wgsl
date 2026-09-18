@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read_write> island_parents: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> island_state: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> wake_flags: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> slept_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> body_admitted: array<u32>;

fn work(index: u32) {
    var state = body_states[index];
    if (state.sleeping != 0u || !body_simulates(body_descs[index])) {
        return;
    }
    let root = atomicLoad(&island_parents[index]);
    let disturbed = (atomicLoad(&island_state[root]) & ISLAND_WAKE) != 0u
        || atomicLoad(&wake_flags[index]) != 0u;
    if (disturbed) {
        state.sleep_timer = 0.0;
    } else {
        state.sleep_timer = state.sleep_timer + params.dt;
        if (state.sleep_timer >= params.sleep_time) {
            freeze_body(&state);
            atomicAdd(&slept_count[0], 1u);
            body_admitted[index] = 0u;
        }
    }
    body_states[index] = state;
}
