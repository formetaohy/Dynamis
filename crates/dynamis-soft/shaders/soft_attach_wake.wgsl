@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> attachments: array<SoftAttachment>;
@group(0) @binding(2) var<storage, read> particles: array<SoftParticle>;
@group(0) @binding(3) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(4) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(5) var<storage, read> row_of_body: array<u32>;
@group(0) @binding(6) var<storage, read_write> bodies: array<SoftBody>;

fn work(index: u32) {
    let attachment = attachments[index];
    if (attachment.particle == NO_SLOT) {
        return;
    }
    let particle = particles[attachment.particle];
    if (particle.owner == NO_BODY || bodies[particle.owner].sleeping == 0u) {
        return;
    }
    let row = rigid_row(attachment.body_id, attachment.generation);
    if (row == NO_BODY) {
        return;
    }
    let body = Body(body_states[row], body_descs[row]);
    if (!body_is_active(body.state, body.desc)) {
        return;
    }
    let anchor = rigid_anchor(body, attachment.local);
    if (length(anchor - particle.position.xyz) > params.slop) {
        atomicStore(&bodies[particle.owner].wake, 1u);
    }
}
