@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read_write> body_activity: array<u32>;
@group(0) @binding(4) var<storage, read_write> active_count: array<atomic<u32>>;

fn work(index: u32) {
    let moving = body_is_active(body_states[index], body_descs[index]);
    body_activity[index] = select(0u, 1u, moving);
    if (moving) {
        atomicStore(&active_count[0], 1u);
    }
}
