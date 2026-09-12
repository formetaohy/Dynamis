@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(3) var<storage, read> constraint_descs: array<ConstraintDescriptor>;
@group(0) @binding(4) var<storage, read> constraint_runtime: array<ConstraintRuntime>;
@group(0) @binding(5) var<storage, read_write> island_parents: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> wake_flags: array<atomic<u32>>;

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn island_link(a: u32, b: u32) {
    atomicMin(&island_parents[a], b);
    atomicMin(&island_parents[b], a);
}

fn work(index: u32) {
    if (constraint_runtime[index].broken != 0u) {
        return;
    }
    let constraint = constraint_descs[index];
    let first = load_body(constraint.a);
    let second = load_body(constraint.b);
    let first_static = body_is_static(first);
    let second_static = body_is_static(second);
    if (first_static && atomicLoad(&wake_flags[constraint.a]) != 0u) {
        atomicOr(&wake_flags[constraint.b], 1u);
    }
    if (second_static && atomicLoad(&wake_flags[constraint.b]) != 0u) {
        atomicOr(&wake_flags[constraint.a], 1u);
    }
    if (!first_static && !second_static && body_is_dynamic(first) && body_is_dynamic(second)) {
        island_link(constraint.a, constraint.b);
    }
}
