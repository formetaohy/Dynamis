@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> first_a: array<u32>;
@group(0) @binding(3) var<storage, read> first_b: array<u32>;
@group(0) @binding(4) var<storage, read> a_bodies: array<u32>;
@group(0) @binding(5) var<storage, read> b_bodies: array<u32>;
@group(0) @binding(6) var<storage, read> b_blocks: array<u32>;
@group(0) @binding(7) var<storage, read> contact_counts: array<u32>;
@group(0) @binding(8) var<storage, read> block_corrections: array<vec4f>;
@group(0) @binding(9) var<storage, read> block_count: array<u32>;
@group(0) @binding(10) var<storage, read_write> resolution: array<vec4f>;
@group(0) @binding(11) var<storage, read_write> contributions: array<atomic<u32>>;

fn work(index: u32) {
    let blocks = contact_counts[index];
    let contributing = atomicLoad(&contributions[index]);
    atomicStore(&contributions[index], 0u);
    if (blocks == 0u || contributing == 0u) {
        return;
    }
    let total = min(block_count[0], arrayLength(&a_bodies));
    let body = body_states[index];
    var correction = vec3f(0.0);
    var start = i32(first_a[index]) - 1;
    if (start >= 0) {
        var end = u32(start);
        while (end < total && a_bodies[end] == index) {
            end = end + 1u;
        }
        for (var i = u32(start); i < end; i = i + 1u) {
            correction = correction + block_corrections[i * 2u].xyz;
        }
    }
    start = i32(first_b[index]) - 1;
    if (start >= 0) {
        var end = u32(start);
        while (end < total && b_bodies[end] == index) {
            end = end + 1u;
        }
        for (var i = u32(start); i < end; i = i + 1u) {
            correction = correction + block_corrections[b_blocks[i] * 2u + 1u].xyz;
        }
    }
    let applied = correction / f32(contributing);
    var updated = body;
    updated.position = body.position + applied;
    resolution[index] = vec4f(resolution[index].xyz + applied, 0.0);
    body_states[index] = updated;
}
