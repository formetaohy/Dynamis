@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read_write> live_bodies: array<u32>;
@group(0) @binding(4) var<storage, read_write> live_count: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> spillover: array<atomic<u32>>;

fn work(index: u32) {
    if (!body_is_active(body_states[index], body_descs[index])) {
        return;
    }
    let slot = atomicAdd(&live_count[0], 1u);
    if (slot < arrayLength(&live_bodies)) {
        live_bodies[slot] = index;
    } else {
        atomicAdd(&spillover[0], 1u);
    }
}
