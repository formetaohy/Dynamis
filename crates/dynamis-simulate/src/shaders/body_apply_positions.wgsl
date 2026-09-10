@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> first_a: array<u32>;
@group(0) @binding(3) var<storage, read> first_b: array<u32>;
@group(0) @binding(4) var<storage, read> a_body: array<u32>;
@group(0) @binding(5) var<storage, read> b_bodies: array<u32>;
@group(0) @binding(6) var<storage, read> b_rows: array<u32>;
@group(0) @binding(7) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> deltas: array<vec4f>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let body_index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (body_index >= params.dynamic_count) {
        return;
    }
    let total = min(atomicLoad(&contact_count[0]), arrayLength(&a_body));
    let body = body_states[body_index];
    var position = vec3f(0.0);
    var links = 0u;
    var start = i32(first_a[body_index]) - 1;
    if (start >= 0) {
        var end = u32(start);
        while (end < total && a_body[end] == body_index) {
            end = end + 1u;
        }
        links = links + end - u32(start);
        for (var i = u32(start); i < end; i = i + 1u) {
            position = position + deltas[i * 4u].xyz;
        }
    }
    start = i32(first_b[body_index]) - 1;
    if (start >= 0) {
        var end = u32(start);
        while (end < total && b_bodies[end] == body_index) {
            end = end + 1u;
        }
        links = links + end - u32(start);
        for (var i = u32(start); i < end; i = i + 1u) {
            position = position + deltas[b_rows[i] * 4u + 1u].xyz;
        }
    }
    let damping = select(1.0, 1.0 / f32(links), links > 1u);
    var updated = body;
    updated.position = body.position + position * damping;
    body_states[body_index] = updated;
}
