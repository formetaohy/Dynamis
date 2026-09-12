@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SoftParticle>;
@group(0) @binding(2) var<storage, read> links: array<SoftLink>;
@group(0) @binding(3) var<storage, read> link_deltas: array<vec4f>;
@group(0) @binding(4) var<storage, read> adjacency: array<u32>;

fn work(index: u32) {
    let particle = particles[index];
    if (particle.owner == NO_BODY) {
        return;
    }
    let weight = particle.prev_position.w;
    if (weight <= 0.0) {
        return;
    }
    var total = vec3f(0.0);
    var count = 0u;
    for (var slot = 0u; slot < particle.neighbour_count; slot = slot + 1u) {
        let link_index = adjacency[particle.neighbour_offset + slot];
        let delta = link_deltas[link_index];
        if (delta.w == 0.0) {
            continue;
        }
        if (links[link_index].first == index) {
            total = total + delta.xyz;
        } else {
            total = total - delta.xyz;
        }
        count = count + 1u;
    }
    if (count == 0u) {
        return;
    }
    var moved = particle;
    moved.position = vec4f(
        particle.position.xyz + total * (weight / f32(count)),
        particle.position.w,
    );
    particles[index] = moved;
}

fn extent() -> u32 {
    return arrayLength(&particles);
}
