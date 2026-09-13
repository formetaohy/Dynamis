@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> particles: array<SoftParticle>;
@group(0) @binding(2) var<storage, read_write> bodies: array<SoftBody>;

fn work(index: u32) {
    let particle = particles[index];
    if (particle.owner == NO_BODY || bodies[particle.owner].sleeping != 0u) {
        return;
    }
    if (length(particle.velocity.xyz) > params.settle_velocity) {
        atomicStore(&bodies[particle.owner].moving, 1u);
    }
}
