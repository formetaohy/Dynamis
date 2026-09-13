@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> particles: array<SoftParticle>;
@group(0) @binding(2) var<storage, read_write> elements: array<SoftElement>;
@group(0) @binding(3) var<storage, read_write> contributions: array<vec4f>;

fn work(index: u32) {
    var element = elements[index];
    let base = index * ELEMENT_PARTICLES;
    for (var slot = 0u; slot < ELEMENT_PARTICLES; slot = slot + 1u) {
        contributions[base + slot] = vec4f(0.0);
    }
    if (element.particles[0] == NO_SLOT || (element.kind & ELEMENT_BROKEN) != 0u) {
        return;
    }
    let kind = element.kind & ELEMENT_KIND_MASK;
    let arity = element_arity(kind);
    var positions: array<vec3f, ELEMENT_PARTICLES>;
    var weights: array<f32, ELEMENT_PARTICLES>;
    var weight = 0.0;
    for (var slot = 0u; slot < arity; slot = slot + 1u) {
        let particle = particles[element.particles[slot]];
        if (particle.owner == NO_BODY) {
            return;
        }
        positions[slot] = particle.position.xyz;
        weights[slot] = particle.prev_position.w;
        weight = weight + weights[slot];
    }
    if (weight <= 0.0) {
        return;
    }
    let sample = element_sample(kind, element.rest, positions);
    if (!sample.valid) {
        return;
    }
    var momentum = 0.0;
    for (var slot = 0u; slot < arity; slot = slot + 1u) {
        momentum = momentum + weights[slot] * dot(sample.gradients[slot], sample.gradients[slot]);
    }
    let compliance = element.compliance / (params.soft_substep_dt * params.soft_substep_dt);
    let denominator = momentum + compliance;
    if (denominator <= 0.0) {
        return;
    }
    let delta = -(sample.value + compliance * element.lambda) / denominator;
    element.lambda = element.lambda + delta;
    for (var slot = 0u; slot < arity; slot = slot + 1u) {
        contributions[base + slot] = vec4f(sample.gradients[slot] * delta, 0.0);
    }
    elements[index] = element;
}
