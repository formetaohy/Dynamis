@group(0) @binding(0) var<storage, read> row_streams: RowStreams;
@group(0) @binding(1) var<storage, read_write> particles: array<SoftParticle>;
@group(0) @binding(2) var<storage, read> edits: array<SoftEdit>;

fn work(index: u32) {
    let edit = edits[index];
    var particle = particles[edit.particle];
    if ((edit.mask & SOFT_EDIT_INVERSE_MASS) != 0u) {
        particle.prev_position = vec4f(particle.prev_position.xyz, edit.inverse_mass);
    }
    if ((edit.mask & SOFT_EDIT_RADIUS) != 0u) {
        particle.position = vec4f(particle.position.xyz, edit.radius);
    }
    if ((edit.mask & SOFT_EDIT_FRICTION) != 0u) {
        particle.velocity = vec4f(particle.velocity.xyz, edit.friction);
    }
    particles[edit.particle] = particle;
}
