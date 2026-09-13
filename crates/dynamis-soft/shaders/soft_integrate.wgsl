@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SoftParticle>;

fn work(index: u32) {
    if (index >= params.particle_count) {
        return;
    }
    var particle = particles[index];
    if (particle.owner == NO_BODY) {
        return;
    }
    let weight = particle.prev_position.w;
    if (weight <= 0.0) {
        particle.prev_position = vec4f(particle.position.xyz, weight);
        particle.velocity = vec4f(0.0, 0.0, 0.0, particle.velocity.w);
        particles[index] = particle;
        return;
    }
    let damping = 1.0 / (1.0 + params.damping * params.soft_substep_dt);
    var velocity = (particle.velocity.xyz + params.gravity.xyz * params.soft_substep_dt) * damping;
    let speed = length(velocity);
    if (speed > params.max_velocity) {
        velocity = velocity * (params.max_velocity / speed);
    }
    particle.velocity = vec4f(velocity, particle.velocity.w);
    particle.prev_position = vec4f(particle.position.xyz, weight);
    particle.position = vec4f(particle.position.xyz + velocity * params.soft_substep_dt, particle.position.w);
    particles[index] = particle;
}

fn extent() -> u32 {
    return arrayLength(&particles);
}
