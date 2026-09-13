@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> particles: array<SoftParticle>;
@group(0) @binding(2) var<storage, read_write> elements: array<SoftElement>;
@group(0) @binding(3) var<storage, read_write> element_deltas: array<f32>;

fn distance_multiplier(element: SoftElement) -> f32 {
    let first = particles[element.particles[0]];
    let second = particles[element.particles[1]];
    if (first.owner == NO_BODY || second.owner == NO_BODY) {
        return 0.0;
    }
    let first_weight = first.prev_position.w;
    let second_weight = second.prev_position.w;
    let weight = first_weight + second_weight;
    if (weight <= 0.0) {
        return 0.0;
    }
    let length = length(second.position.xyz - first.position.xyz);
    if (length < 1e-8) {
        return 0.0;
    }
    let compliance = element.compliance / (params.soft_substep_dt * params.soft_substep_dt);
    return (-(length - element.rest) - compliance * element.lambda) / (weight + compliance);
}

fn work(index: u32) {
    if (index >= params.element_count) {
        return;
    }
    var element = elements[index];
    element_deltas[index] = 0.0;
    if (element.particles[0] == NO_SLOT) {
        return;
    }
    let delta = distance_multiplier(element);
    element.lambda = element.lambda + delta;
    element_deltas[index] = delta;
    elements[index] = element;
}

fn extent() -> u32 {
    return arrayLength(&elements);
}
