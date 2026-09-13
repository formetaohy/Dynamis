@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SoftParticle>;
@group(0) @binding(2) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(3) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(4) var<storage, read> contacts: array<SoftContact>;
@group(0) @binding(5) var<storage, read_write> reactions: array<atomic<u32>>;

const REACTION_SCALE: f32 = 65536.0;
const REACTION_WORDS: u32 = 8u;

fn load_body(row: u32) -> Body {
    return Body(body_states[row], body_descs[row]);
}

fn fixed_word(value: f32) -> u32 {
    return u32(i32(clamp(value * REACTION_SCALE, -2.0e9, 2.0e9)));
}

fn accumulate_reaction(row: u32, shift: vec3f, spin: vec3f) {
    let base = row * REACTION_WORDS;
    atomicAdd(&reactions[base], fixed_word(shift.x));
    atomicAdd(&reactions[base + 1u], fixed_word(shift.y));
    atomicAdd(&reactions[base + 2u], fixed_word(shift.z));
    atomicAdd(&reactions[base + 4u], fixed_word(spin.x));
    atomicAdd(&reactions[base + 5u], fixed_word(spin.y));
    atomicAdd(&reactions[base + 6u], fixed_word(spin.z));
}

fn apply_friction(
    particle: ptr<function, SoftParticle>,
    normal: vec3f,
    depth: f32,
    friction: f32,
) {
    let velocity = (*particle).velocity.xyz;
    let tangential = velocity - normal * dot(velocity, normal);
    let slide = length(tangential) * params.soft_substep_dt;
    if (slide <= 1e-6) {
        return;
    }
    let magnitude = min(slide, sqrt(max(friction, 0.0)) * depth);
    (*particle).position = vec4f(
        (*particle).position.xyz - normalize(tangential) * magnitude,
        (*particle).position.w,
    );
}

fn hold_contact_velocity(particle: ptr<function, SoftParticle>, normal: vec3f) {
    let velocity = (*particle).velocity.xyz;
    let closing = dot(velocity, normal);
    if (closing < 0.0) {
        (*particle).velocity = vec4f(velocity - normal * closing, (*particle).velocity.w);
    }
}

fn work(index: u32) {
    let contact = contacts[index];
    if (contact.partner == NO_SLOT) {
        return;
    }
    var particle = particles[index];
    if (particle.owner == NO_BODY) {
        return;
    }
    let weight = particle.prev_position.w;
    let radius = particle.position.w;
    let normal = contact.normal;
    let depth = contact.depth;
    let friction = max(particle.velocity.w, 0.0) * contact.friction;
    if (contact.kind == ENTRY_KIND_COLLIDER) {
        let body = load_body(contact.group);
        var shifted = particle.position.xyz + normal * depth;
        if (body.desc.inverse_mass > 0.0) {
            let ra = contact.point - body_com(body);
            let rn = cross(ra, normal);
            let body_weight = body.desc.inverse_mass;
            let k = weight + body_weight + dot(rn, apply_inverse_inertia(body, rn));
            if (k > 0.0) {
                let lambda = depth / k;
                shifted = particle.position.xyz + normal * (lambda * weight);
                accumulate_reaction(
                    contact.group,
                    -normal * (lambda * body_weight),
                    -apply_inverse_inertia(body, rn) * lambda,
                );
            }
        }
        if (weight > 0.0) {
            particle.position = vec4f(shifted, radius);
            hold_contact_velocity(&particle, normal);
            apply_friction(&particle, normal, depth, friction);
            particles[index] = particle;
        }
        return;
    }
    if (weight <= 0.0) {
        return;
    }
    let total = weight + contact.partner_inverse_mass;
    if (total <= 0.0) {
        return;
    }
    particle.position = vec4f(
        particle.position.xyz - normal * (depth * weight / total),
        radius,
    );
    hold_contact_velocity(&particle, normal);
    apply_friction(&particle, normal, depth, friction);
    particles[index] = particle;
}
