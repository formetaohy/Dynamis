@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> attachments: array<SoftAttachment>;
@group(0) @binding(2) var<storage, read_write> particles: array<SoftParticle>;
@group(0) @binding(3) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(4) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(5) var<storage, read> row_of_body: array<u32>;
@group(0) @binding(6) var<storage, read_write> reactions: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> bodies: array<SoftBody>;

fn work(index: u32) {
    let attachment = attachments[index];
    if (attachment.particle == NO_SLOT) {
        return;
    }
    var particle = particles[attachment.particle];
    if (particle.owner == NO_BODY || bodies[particle.owner].sleeping != 0u) {
        return;
    }
    let row = rigid_row(attachment.body_id, attachment.generation);
    if (row == NO_BODY) {
        return;
    }
    let body = Body(body_states[row], body_descs[row]);
    let anchor = rigid_anchor(body, attachment.local);
    let offset = anchor - particle.position.xyz;
    let error = length(offset);
    if (error <= 1e-6) {
        return;
    }
    let axis = offset / error;
    let weight = particle.prev_position.w;
    var anchor_weight = 0.0;
    var lever = vec3f(0.0);
    if (!body_is_inert(body)) {
        lever = cross(anchor - body_com(body), axis);
        anchor_weight = body.desc.inverse_mass + dot(lever, apply_inverse_inertia(body, lever));
    }
    let total = weight + anchor_weight;
    if (total <= 0.0) {
        return;
    }
    let lambda = error / total;
    particle.position = vec4f(
        particle.position.xyz + axis * (lambda * weight),
        particle.position.w,
    );
    particle.velocity = vec4f(
        (particle.position.xyz - particle.prev_position.xyz) / params.soft_substep_dt,
        particle.velocity.w,
    );
    particles[attachment.particle] = particle;
    if (anchor_weight > 0.0) {
        accumulate_reaction(
            row,
            -axis * (lambda * body.desc.inverse_mass),
            -apply_inverse_inertia(body, lever) * lambda,
        );
    }
}
