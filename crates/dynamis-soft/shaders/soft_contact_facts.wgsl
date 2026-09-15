@group(0) @binding(0) var<storage, read> contacts: array<SoftContact>;
@group(0) @binding(1) var<storage, read> bodies: array<SoftBody>;
@group(0) @binding(2) var<storage, read> colliders: array<Collider>;
@group(0) @binding(3) var<storage, read_write> particles: array<SoftParticle>;
@group(0) @binding(4) var<storage, read_write> events: array<ContactEvent>;
@group(0) @binding(5) var<storage, read_write> event_count: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> spillover: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params: StepParams;

fn announcements_of(owner: u32) -> u32 {
    if (owner == NO_BODY) {
        return 0u;
    }
    return bodies[owner].events;
}

fn partner_announcements(fact: SoftFact) -> u32 {
    let modes = EVENT_MODE_BEGIN_END | EVENT_MODE_PERSIST;
    let kind = fact.scene_target >> ENTRY_KIND_SHIFT;
    if (kind == ENTRY_KIND_COLLIDER) {
        let collider = fact.scene_target & ENTRY_INDEX_MASK;
        if (collider < arrayLength(&colliders)) {
            return colliders[collider].flags & modes;
        }
        return 0u;
    }
    return announcements_of(fact.id) & modes;
}

fn announceable(record: SoftContact) -> SoftFact {
    if (record.sensor.scene_target != NO_SLOT) {
        return record.sensor;
    }
    return record.contact;
}

fn announced(record: SoftContact, index: u32, step: u32) {
    var particle = particles[index];
    if (particle.announcement.step == step) {
        return;
    }
    let owner_modes = announcements_of(particle.owner);
    let fact = announceable(record);
    let holds_fact = fact.scene_target != NO_SLOT;
    let sensor = select(0u, 1u, record.sensor.scene_target != NO_SLOT);
    var modes = 0u;
    if (holds_fact) {
        modes = owner_modes & partner_announcements(fact);
    }
    let holds_announced = holds_fact
        && particle.announcement.announced != 0u
        && particle.announcement.scene_target == fact.scene_target
        && particle.announcement.id == fact.id
        && particle.announcement.generation == fact.generation;
    if (!holds_announced && particle.announcement.announced != 0u) {
        if ((owner_modes & EVENT_MODE_BEGIN_END) != 0u) {
            emit_contact_fact(
                EVENT_END,
                particle.announcement.sensor,
                scene_target_of(ENTRY_KIND_PARTICLE, index),
                particle.owner,
                particle.generation,
                particle.announcement.scene_target,
                particle.announcement.id,
                particle.announcement.generation,
                particle.position.xyz,
                vec3f(0.0, 1.0, 0.0),
            );
        }
        particle.announcement.announced = 0u;
    }
    if (!holds_announced) {
        if (holds_fact && (modes & EVENT_MODE_BEGIN_END) != 0u) {
            emit_contact_fact(
                EVENT_BEGIN,
                sensor,
                scene_target_of(ENTRY_KIND_PARTICLE, index),
                particle.owner,
                particle.generation,
                fact.scene_target,
                fact.id,
                fact.generation,
                record.point,
                record.normal,
            );
            particle.announcement.scene_target = fact.scene_target;
            particle.announcement.id = fact.id;
            particle.announcement.generation = fact.generation;
            particle.announcement.sensor = sensor;
            particle.announcement.announced = 1u;
        }
    } else if ((modes & EVENT_MODE_PERSIST) != 0u) {
        emit_contact_fact(
            EVENT_PERSIST,
            sensor,
            scene_target_of(ENTRY_KIND_PARTICLE, index),
            particle.owner,
            particle.generation,
            fact.scene_target,
            fact.id,
            fact.generation,
            record.point,
            record.normal,
        );
    }
    particle.announcement.step = step;
    particles[index] = particle;
}

fn work(index: u32) {
    let particle = particles[index];
    if (particle.owner == NO_BODY || bodies[particle.owner].sleeping != 0u) {
        return;
    }
    announced(contacts[index], index, counter_load(COUNTER_STEP));
}
