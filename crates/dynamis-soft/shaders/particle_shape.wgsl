fn particle_tight_bounds(particle: SoftParticle) -> Aabb {
    let radius = particle.position.w;
    var box: Aabb;
    box.min = particle.position.xyz - vec3f(radius);
    box.max = particle.position.xyz + vec3f(radius);
    return box;
}

fn particle_swept_bounds(particle: SoftParticle, dt: f32, gravity: vec3f) -> Aabb {
    let tight = particle_tight_bounds(particle);
    let predicted = particle.position.xyz + particle.velocity.xyz * dt + gravity * dt * dt;
    let travel = abs(predicted - particle.position.xyz);
    var box: Aabb;
    box.min = tight.min - travel;
    box.max = tight.max + travel;
    return box;
}
