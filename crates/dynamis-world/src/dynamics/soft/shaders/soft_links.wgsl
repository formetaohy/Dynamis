@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> particles: array<SoftParticle>;
@group(0) @binding(2) var<storage, read> links: array<SoftLink>;
@group(0) @binding(3) var<storage, read_write> link_deltas: array<vec4f>;

fn work(index: u32) {
    link_deltas[index] = vec4f(0.0);
    let link = links[index];
    if (link.first == NO_SLOT) {
        return;
    }
    let first = particles[link.first];
    let second = particles[link.second];
    if (first.owner == NO_BODY || second.owner == NO_BODY) {
        return;
    }
    let first_weight = first.prev_position.w;
    let second_weight = second.prev_position.w;
    let weight = first_weight + second_weight;
    if (weight <= 0.0) {
        return;
    }
    let offset = second.position.xyz - first.position.xyz;
    let length = length(offset);
    if (length < 1e-8) {
        return;
    }
    let compliance = params.soft_compliance / (params.dt * params.dt);
    let lambda = (length - link.rest) / (weight + compliance);
    link_deltas[index] = vec4f(offset / length * lambda, 1.0);
}

fn extent() -> u32 {
    return arrayLength(&links);
}
