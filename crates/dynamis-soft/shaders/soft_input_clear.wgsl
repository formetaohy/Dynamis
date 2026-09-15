@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<SoftBody>;

fn work(index: u32) {
    bodies[index].acceleration = vec3f(0.0);
}
