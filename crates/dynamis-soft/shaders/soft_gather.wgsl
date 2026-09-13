@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SoftParticle>;
@group(0) @binding(2) var<storage, read> contributions: array<vec4f>;
@group(0) @binding(3) var<storage, read> adjacency: array<u32>;
@group(0) @binding(4) var<storage, read> elements: array<SoftElement>;
@group(0) @binding(5) var<storage, read> bodies: array<SoftBody>;

fn work(index: u32) {
    let particle = particles[index];
    if (particle.owner == NO_BODY || bodies[particle.owner].sleeping != 0u) {
        return;
    }
    let weight = particle.prev_position.w;
    if (weight <= 0.0 || particle.neighbour_count == 0u) {
        return;
    }
    var total = vec3f(0.0);
    var live = 0u;
    for (var slot = 0u; slot < particle.neighbour_count; slot = slot + 1u) {
        let entry = adjacency[particle.neighbour_offset + slot];
        if ((elements[entry >> ELEMENT_ROLE_BITS].kind & ELEMENT_BROKEN) != 0u) {
            continue;
        }
        let contribution = (entry >> ELEMENT_ROLE_BITS) * ELEMENT_PARTICLES
            + (entry & ELEMENT_ROLE_MASK);
        total = total + contributions[contribution].xyz;
        live = live + 1u;
    }
    if (live == 0u) {
        return;
    }
    var moved = particle;
    moved.position = vec4f(
        particle.position.xyz + total * (weight / f32(live)),
        particle.position.w,
    );
    moved.velocity = vec4f(
        (moved.position.xyz - particle.prev_position.xyz) / params.soft_substep_dt,
        particle.velocity.w,
    );
    particles[index] = moved;
}
