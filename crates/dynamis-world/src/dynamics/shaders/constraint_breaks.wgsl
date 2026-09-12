@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(2) var<storage, read_write> constraint_breaks: array<BrokenConstraint>;
@group(0) @binding(3) var<storage, read_write> break_count: array<atomic<u32>>;

fn work(index: u32) {
    let runtime = constraint_runtime[index];
    if (runtime.broken == 0u) {
        return;
    }
    let segment = arrayLength(&constraint_breaks) / EVENT_SLOTS;
    let slot = atomicAdd(&break_count[0], 1u);
    if (slot >= segment) {
        return;
    }
    constraint_breaks[params.event_slot * segment + slot] =
        BrokenConstraint(runtime.constraint_id, runtime.generation);
}
