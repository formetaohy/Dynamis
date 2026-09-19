@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SoftParticle>;
@group(0) @binding(2) var<storage, read> bodies: array<SoftBody>;
@group(0) @binding(3) var<storage, read> fields: array<Field>;

fn work(index: u32) {
    var particle = particles[index];
    if (particle.owner == NO_BODY) {
        return;
    }
    if (bodies[particle.owner].sleeping != 0u) {
        particle.prev_position = vec4f(particle.position.xyz, particle.prev_position.w);
        particle.velocity = vec4f(0.0, 0.0, 0.0, particle.velocity.w);
        particles[index] = particle;
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
    var acceleration = params.gravity.xyz + bodies[particle.owner].acceleration;
    var velocity = particle.velocity.xyz;
    let reach = vec2u(
        bodies[particle.owner].collision_group,
        bodies[particle.owner].collision_mask,
    );
    let field_count = min(params.field_count, arrayLength(&fields));
    for (var field = 0u; field < field_count; field = field + 1u) {
        let force = field_force(
            fields[field],
            particle.position.xyz,
            velocity,
            params.gravity.xyz,
            reach,
            params.soft_substep_dt,
        );
        acceleration = acceleration + force.acceleration;
        velocity = velocity + force.velocity;
    }
    velocity = (velocity + acceleration * params.soft_substep_dt) * damping;
    let speed = length(velocity);
    if (speed > params.max_velocity) {
        velocity = velocity * (params.max_velocity / speed);
    }
    particle.velocity = vec4f(velocity, particle.velocity.w);
    particle.prev_position = vec4f(particle.position.xyz, weight);
    particle.position = vec4f(particle.position.xyz + velocity * params.soft_substep_dt, particle.position.w);
    particles[index] = particle;
}
