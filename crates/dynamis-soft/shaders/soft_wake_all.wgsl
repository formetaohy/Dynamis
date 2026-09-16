@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<SoftBody>;

fn work(index: u32) {
    if (params.wake_all == 0u) {
        return;
    }
    atomicStore(&bodies[index].wake, 1u);
}
