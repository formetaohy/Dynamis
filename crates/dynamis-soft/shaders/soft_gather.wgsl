@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SoftParticle>;
@group(0) @binding(2) var<storage, read> contributions: array<vec4f>;
@group(0) @binding(3) var<storage, read> adjacency: array<u32>;

fn work(index: u32) {
    let particle = particles[index];
    if (particle.owner == NO_BODY) {
        return;
    }
    let weight = particle.prev_position.w;
    if (weight <= 0.0 || particle.neighbour_count == 0u) {
        return;
    }
    var total = vec3f(0.0);
    for (var slot = 0u; slot < particle.neighbour_count; slot = slot + 1u) {
        let entry = adjacency[particle.neighbour_offset + slot];
        let contribution = (entry >> ELEMENT_ROLE_BITS) * ELEMENT_PARTICLES
            + (entry & ELEMENT_ROLE_MASK);
        total = total + contributions[contribution].xyz;
    }
    var moved = particle;
    moved.position = vec4f(
        particle.position.xyz + total * (weight / f32(particle.neighbour_count)),
        particle.position.w,
    );
    moved.velocity = vec4f(
        (moved.position.xyz - particle.prev_position.xyz) / params.soft_substep_dt,
        particle.velocity.w,
    );
    particles[index] = moved;
}
