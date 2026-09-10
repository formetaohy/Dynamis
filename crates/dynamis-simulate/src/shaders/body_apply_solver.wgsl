@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> body_states: array<BodyState>;
@group(0) @binding(2) var<storage, read> first_a: array<u32>;
@group(0) @binding(3) var<storage, read> first_b: array<u32>;
@group(0) @binding(4) var<storage, read> a_body: array<u32>;
@group(0) @binding(5) var<storage, read> b_bodies: array<u32>;
@group(0) @binding(6) var<storage, read> b_rows: array<u32>;
@group(0) @binding(7) var<storage, read> constraint_a_first: array<u32>;
@group(0) @binding(8) var<storage, read> constraint_b_first: array<u32>;
@group(0) @binding(9) var<storage, read> constraint_a_bodies: array<u32>;
@group(0) @binding(10) var<storage, read> constraint_a_rows: array<u32>;
@group(0) @binding(11) var<storage, read> constraint_b_bodies: array<u32>;
@group(0) @binding(12) var<storage, read> constraint_b_rows: array<u32>;
@group(0) @binding(13) var<storage, read_write> contact_count: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read> deltas: array<vec4f>;
@group(0) @binding(15) var<storage, read> constraint_deltas: array<vec4f>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let body_index = gid.y * (WORKGROUPS_PER_ROW * WORKGROUP_SIZE) + gid.x;
    if (body_index >= params.dynamic_count) {
        return;
    }
    let total = min(atomicLoad(&contact_count[0]), arrayLength(&a_body));
    let constraint_total = params.constraint_count;
    let body = body_states[body_index];
    var velocity = vec3f(0.0);
    var spin = vec3f(0.0);
    var links = 0u;
    var start = i32(first_a[body_index]) - 1;
    if (start >= 0) {
        var end = u32(start);
        while (end < total && a_body[end] == body_index) {
            end = end + 1u;
        }
        links = links + end - u32(start);
        for (var i = u32(start); i < end; i = i + 1u) {
            velocity = velocity + deltas[i * 4u].xyz;
            spin = spin + deltas[i * 4u + 1u].xyz;
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
            let value = b_rows[i];
            velocity = velocity + deltas[value * 4u + 2u].xyz;
            spin = spin + deltas[value * 4u + 3u].xyz;
        }
    }
    start = i32(constraint_a_first[body_index]) - 1;
    if (start >= 0) {
        var end = u32(start);
        while (end < constraint_total && constraint_a_bodies[end] == body_index) {
            end = end + 1u;
        }
        links = links + end - u32(start);
        for (var i = u32(start); i < end; i = i + 1u) {
            let value = constraint_a_rows[i];
            velocity = velocity + constraint_deltas[value * 4u].xyz;
            spin = spin + constraint_deltas[value * 4u + 1u].xyz;
        }
    }
    start = i32(constraint_b_first[body_index]) - 1;
    if (start >= 0) {
        var end = u32(start);
        while (end < constraint_total && constraint_b_bodies[end] == body_index) {
            end = end + 1u;
        }
        links = links + end - u32(start);
        for (var i = u32(start); i < end; i = i + 1u) {
            let value = constraint_b_rows[i];
            velocity = velocity + constraint_deltas[value * 4u + 2u].xyz;
            spin = spin + constraint_deltas[value * 4u + 3u].xyz;
        }
    }

    let damping = select(1.0, (1.0 - params.tempering) / f32(links) + params.tempering, links > 1u);
    var updated = body;
    updated.velocity = body.velocity + velocity * damping;
    updated.angular_velocity = body.angular_velocity + spin * damping;
    body_states[body_index] = updated;
}
