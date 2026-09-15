@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<SoftBody>;
@group(0) @binding(2) var<storage, read> edits: array<SoftBodyEdit>;

fn work(index: u32) {
    let edit = edits[index];
    let owner = edit.owner;
    if ((edit.mask & SOFT_BODY_EDIT_ACCELERATION) != 0u) {
        bodies[owner].acceleration = bodies[owner].acceleration + edit.acceleration;
    }
    if ((edit.mask & SOFT_BODY_EDIT_WAKE) != 0u) {
        bodies[owner].sleep_timer = 0.0;
        bodies[owner].sleeping = 0u;
    }
}
