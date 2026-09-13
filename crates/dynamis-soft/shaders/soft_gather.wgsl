@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SoftParticle>;
@group(0) @binding(2) var<storage, read> elements: array<SoftElement>;
@group(0) @binding(3) var<storage, read> element_deltas: array<f32>;
@group(0) @binding(4) var<storage, read> adjacency: array<u32>;

fn element_gradient(element: SoftElement, role: u32) -> vec3f {
    let direction = sign_normalize(
        particles[element.particles[1]].position.xyz - particles[element.particles[0]].position.xyz,
    );
    return select(-direction, direction, role == 1u);
}

fn work(index: u32) {
    if (index >= params.particle_count) {
        return;
    }
    let particle = particles[index];
    if (particle.owner == NO_BODY) {
        return;
    }
    let weight = particle.prev_position.w;
    if (weight <= 0.0) {
        return;
    }
    if (particle.neighbour_count == 0u) {
        return;
    }
    var total = vec3f(0.0);
    for (var slot = 0u; slot < particle.neighbour_count; slot = slot + 1u) {
        let entry = adjacency[particle.neighbour_offset + slot];
        let element = elements[entry >> 1u];
        total = total + element_gradient(element, entry & 1u) * element_deltas[entry >> 1u];
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

fn extent() -> u32 {
    return arrayLength(&particles);
}
