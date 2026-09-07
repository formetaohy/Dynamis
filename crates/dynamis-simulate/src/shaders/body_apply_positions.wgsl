@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> bodies: array<RigidBody>;
@group(0) @binding(2) var<storage, read> contact_first_a: array<u32>;
@group(0) @binding(3) var<storage, read> contact_first_b: array<u32>;
@group(0) @binding(4) var<storage, read> contact_a_body: array<u32>;
@group(0) @binding(5) var<storage, read> contact_keys_b: array<u32>;
@group(0) @binding(6) var<storage, read> contact_values_b: array<u32>;
@group(0) @binding(7) var<storage, read> contact_count: array<u32>;
@group(0) @binding(8) var<storage, read> contact_deltas: array<vec4f>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let body_index = gid.x;
    if (body_index >= params.body_count) {
        return;
    }
    let total = contact_count[0];
    let body = bodies[body_index];
    var position = vec3f(0.0);
    var links = 0u;
    var start = i32(contact_first_a[body_index]) - 1;
    if (start >= 0) {
        var end = u32(start);
        while (end < total && contact_a_body[end] == body_index) {
            end = end + 1u;
        }
        links = links + end - u32(start);
        for (var i = u32(start); i < end; i = i + 1u) {
            position = position + contact_deltas[i * 4u].xyz;
        }
    }
    start = i32(contact_first_b[body_index]) - 1;
    if (start >= 0) {
        var end = u32(start);
        while (end < total && contact_keys_b[end] == body_index) {
            end = end + 1u;
        }
        links = links + end - u32(start);
        for (var i = u32(start); i < end; i = i + 1u) {
            position = position + contact_deltas[contact_values_b[i] * 4u + 1u].xyz;
        }
    }
    let damping = select(1.0, 1.0 / f32(links), links > 1u);
    var updated = body;
    updated.position = body.position + position * damping;
    bodies[body_index] = updated;
}
