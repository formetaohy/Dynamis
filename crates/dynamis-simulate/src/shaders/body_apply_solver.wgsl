@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> contact_first_a: array<u32>;
@group(0) @binding(3) var<storage, read> contact_first_b: array<u32>;
@group(0) @binding(4) var<storage, read> contact_a_body: array<u32>;
@group(0) @binding(5) var<storage, read> contact_keys_b: array<u32>;
@group(0) @binding(6) var<storage, read> contact_values_b: array<u32>;
@group(0) @binding(7) var<storage, read> constraint_first_a: array<u32>;
@group(0) @binding(8) var<storage, read> constraint_first_b: array<u32>;
@group(0) @binding(9) var<storage, read> constraint_keys_a: array<u32>;
@group(0) @binding(10) var<storage, read> constraint_values_a: array<u32>;
@group(0) @binding(11) var<storage, read> constraint_keys_b: array<u32>;
@group(0) @binding(12) var<storage, read> constraint_values_b: array<u32>;
@group(0) @binding(13) var<storage, read> contact_count: array<u32>;
@group(0) @binding(14) var<storage, read> contact_deltas: array<vec4f>;
@group(0) @binding(15) var<storage, read> constraint_deltas: array<vec4f>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let body_index = gid.x;
    if (body_index >= params.dynamic_count) {
        return;
    }
    let total = contact_count[0];
    let constraint_total = params.constraint_count;
    let body = body_states[body_index];
    var velocity = vec3f(0.0);
    var spin = vec3f(0.0);
    var links = 0u;
    var start = i32(contact_first_a[body_index]) - 1;
    if (start >= 0) {
        var end = u32(start);
        while (end < total && contact_a_body[end] == body_index) {
            end = end + 1u;
        }
        links = links + end - u32(start);
        for (var i = u32(start); i < end; i = i + 1u) {
            velocity = velocity + contact_deltas[i * 4u].xyz;
            spin = spin + contact_deltas[i * 4u + 1u].xyz;
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
            let value = contact_values_b[i];
            velocity = velocity + contact_deltas[value * 4u + 2u].xyz;
            spin = spin + contact_deltas[value * 4u + 3u].xyz;
        }
    }
    start = i32(constraint_first_a[body_index]) - 1;
    if (start >= 0) {
        var end = u32(start);
        while (end < constraint_total && constraint_keys_a[end] == body_index) {
            end = end + 1u;
        }
        links = links + end - u32(start);
        for (var i = u32(start); i < end; i = i + 1u) {
            let value = constraint_values_a[i];
            velocity = velocity + constraint_deltas[value * 4u].xyz;
            spin = spin + constraint_deltas[value * 4u + 1u].xyz;
        }
    }
    start = i32(constraint_first_b[body_index]) - 1;
    if (start >= 0) {
        var end = u32(start);
        while (end < constraint_total && constraint_keys_b[end] == body_index) {
            end = end + 1u;
        }
        links = links + end - u32(start);
        for (var i = u32(start); i < end; i = i + 1u) {
            let value = constraint_values_b[i];
            velocity = velocity + constraint_deltas[value * 4u + 2u].xyz;
            spin = spin + constraint_deltas[value * 4u + 3u].xyz;
        }
    }
    let damping = select(1.0, 1.0 / f32(links), links > 1u);
    var updated = body;
    updated.velocity = body.velocity + velocity * damping;
    updated.angular_velocity = body.angular_velocity + spin * damping;
    body_states[body_index] = updated;
}
