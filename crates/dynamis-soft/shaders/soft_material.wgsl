@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> particles: array<SoftParticle>;
@group(0) @binding(2) var<storage, read_write> elements: array<SoftElement>;

fn work(index: u32) {
    var element = elements[index];
    if (element.particles[0] == NO_SLOT || (element.kind & ELEMENT_BROKEN) != 0u) {
        return;
    }
    let kind = element.kind & ELEMENT_KIND_MASK;
    let arity = element_arity(kind);
    var positions: array<vec3f, ELEMENT_PARTICLES>;
    for (var slot = 0u; slot < arity; slot = slot + 1u) {
        let particle = particles[element.particles[slot]];
        if (particle.owner == NO_BODY) {
            return;
        }
        positions[slot] = particle.position.xyz;
    }
    let sample = element_sample(kind, element.rest, positions);
    if (!sample.valid) {
        return;
    }
    let strain = element_strain(kind, element.rest, sample.value);
    if (strain > element.yield_strain) {
        element.rest = element.rest + sample.value * element.plastic_flow;
    }
    if (strain > element.break_strain) {
        element.kind = element.kind | ELEMENT_BROKEN;
    }
    elements[index] = element;
}
