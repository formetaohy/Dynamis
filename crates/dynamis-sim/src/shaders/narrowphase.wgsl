@group(0) @binding(0) var<storage, read> bodies: array<RigidBody>;
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;
@group(0) @binding(2) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(3) var<storage, read_write> contact_count: atomic<u32>;
@group(0) @binding(4) var<storage, read> pair_count: atomic<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index >= atomicLoad(&pair_count)) {
        return;
    }
    let pair = pairs[index];
    let first = bodies[pair.a];
    let second = bodies[pair.b];
    let delta = second.position - first.position;
    let distance_sq = dot(delta, delta);
    let radius_sum = first.radius + second.radius;
    if (distance_sq >= radius_sum * radius_sum) {
        return;
    }
    let distance = sqrt(distance_sq);
    var normal = vec3f(0.0, 1.0, 0.0);
    if (distance > 1e-6) {
        normal = delta / distance;
    }
    let depth = radius_sum - distance;
    let slot = atomicAdd(&contact_count, 1u);
    if (slot < arrayLength(&contacts)) {
        contacts[slot] = Contact(pair.a, pair.b, depth, 0.0, normal);
    }
}
