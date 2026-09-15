@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read> particles: array<SoftParticle>;
@group(0) @binding(6) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(7) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(8) var<storage, read> colliders: array<Collider>;
@group(0) @binding(9) var<storage, read> elements: array<SoftElement>;
@group(0) @binding(10) var<storage, read> adjacency: array<u32>;
@group(0) @binding(11) var<storage, read_write> contacts: array<SoftContact>;
@group(0) @binding(12) var<storage, read> bodies: array<SoftBody>;

const UNREACHED: f32 = 3.402823466e38;

struct ParticleContact {
    separation: f32,
    normal: vec3f,
    point: vec3f,
    kind: u32,
    partner: u32,
    group: u32,
    id: u32,
    generation: u32,
    friction: f32,
    sensor: u32,
}

fn no_contact() -> ParticleContact {
    var contact: ParticleContact;
    contact.separation = UNREACHED;
    contact.normal = vec3f(0.0, 1.0, 0.0);
    contact.point = vec3f(0.0);
    contact.kind = ENTRY_KIND_COLLIDER;
    contact.partner = NO_SLOT;
    contact.group = NO_BODY;
    contact.id = NO_BODY;
    contact.generation = 0u;
    contact.friction = 0.0;
    contact.sensor = 0u;
    return contact;
}

fn contacts_partner(contact: ParticleContact) -> bool {
    return contact.partner != NO_SLOT;
}

fn contact_touches(contact: ParticleContact) -> bool {
    return contacts_partner(contact) && contact.separation < 0.0;
}

fn sphere_probe(center: vec3f, radius: f32) -> WorldShape {
    var world: WorldShape;
    world.kind = SHAPE_SPHERE;
    world.radius = radius;
    world.half_height = 0.0;
    world.center = center;
    world.half_extents = vec3f(0.0);
    world.rotation = vec4f(0.0, 0.0, 0.0, 1.0);
    world.source = 0u;
    world.scale = vec3f(1.0);
    return world;
}

fn particle_contact(
    world: WorldShape,
    center: vec3f,
    radius: f32,
    out_triangle: ptr<function, u32>,
) -> ParticleContact {
    var contact = no_contact();
    if (world.kind == SHAPE_PLANE) {
        let normal = plane_normal(world);
        contact.separation = dot(center - world.center, normal) - radius;
        contact.normal = normal;
        contact.point = center - normal * radius;
        return contact;
    }
    let probe = sphere_probe(center, radius);
    if (world.kind == SHAPE_MESH || world.kind == SHAPE_HEIGHTFIELD) {
        let closest = scene_convex_closest(world, probe, out_triangle);
        contact.separation = closest.distance;
        contact.normal = closest.normal;
        contact.point = closest.point_a;
        return contact;
    }
    let hit = convex_hit(probe, world);
    contact.separation = hit.distance;
    contact.normal = -hit.normal;
    contact.point = hit.point;
    return contact;
}

fn contact_precedes(candidate: ParticleContact, held: ParticleContact) -> bool {
    if (candidate.separation < held.separation) {
        return true;
    }
    return candidate.separation == held.separation && candidate.partner < held.partner;
}

fn directly_linked(first: u32, second: u32) -> bool {
    let particle = particles[first];
    for (var slot = 0u; slot < particle.neighbour_count; slot = slot + 1u) {
        let element = elements[adjacency[particle.neighbour_offset + slot] >> ELEMENT_ROLE_BITS];
        if ((element.kind & ELEMENT_BROKEN) != 0u) {
            continue;
        }
        for (var role = 0u; role < ELEMENT_PARTICLES; role = role + 1u) {
            if (element.particles[role] == second) {
                return true;
            }
        }
    }
    return false;
}

fn load_body(slot: u32) -> Body {
    return Body(body_states[slot], body_descs[slot]);
}

fn visit_entry(
    node: u32,
    box: Aabb,
    cell_size: f32,
    center: vec3f,
    radius: f32,
    self_index: u32,
    self_owner: u32,
    solid: ptr<function, ParticleContact>,
    contact: ptr<function, ParticleContact>,
    sensor: ptr<function, ParticleContact>,
) {
    if (!aabb_overlaps(entry_box(node), box)) {
        return;
    }
    if (!reach_holds(node, box, cell_size)) {
        return;
    }
    let info = entries[node].info;
    if (entry_kind(info) == ENTRY_KIND_COLLIDER) {
        let collider = colliders[entry_index(info)];
        if (collider.kind == SHAPE_NONE) {
            return;
        }
        let body_slot = entry_group(node);
        if (!filters_intersect(collider_filter(load_body(body_slot), collider), owner_filter(self_owner))) {
            return;
        }
        var triangle = NO_TRIANGLE;
        var probe = particle_contact(world_collider(body_states[body_slot], collider), center, radius, &triangle);
        probe.kind = ENTRY_KIND_COLLIDER;
        probe.partner = entry_index(info);
        probe.group = body_slot;
        probe.id = body_states[body_slot].body_id;
        probe.generation = body_states[body_slot].generation;
        probe.friction = surface_material(collider, triangle).friction;
        probe.sensor = select(0u, 1u, (collider.flags & COLLIDER_SENSOR) != 0u);
        if (probe.sensor != 0u) {
            if (contact_precedes(probe, *sensor)) {
                *sensor = probe;
            }
            return;
        }
        if (contact_precedes(probe, *solid)) {
            *solid = probe;
        }
        if (contact_precedes(probe, *contact)) {
            *contact = probe;
        }
        return;
    }
    let other_index = entry_index(info);
    if (other_index == self_index) {
        return;
    }
    let other = particles[other_index];
    if (other.owner == NO_BODY) {
        return;
    }
    if (other.owner == self_owner && directly_linked(self_index, other_index)) {
        return;
    }
    if (!filters_intersect(owner_filter(other.owner), owner_filter(self_owner))) {
        return;
    }
    let delta = other.position.xyz - center;
    let distance = length(delta);
    let separation = distance - (radius + other.position.w);
    if (separation >= 0.0) {
        return;
    }
    var probe = no_contact();
    probe.separation = separation;
    probe.normal = sign_normalize(delta);
    probe.point = center + probe.normal * (radius - separation * 0.5);
    probe.kind = ENTRY_KIND_PARTICLE;
    probe.partner = other_index;
    probe.group = other.owner;
    probe.id = other.owner;
    probe.generation = other.generation;
    probe.friction = other.velocity.w;
    if (contact_precedes(probe, *solid)) {
        *solid = probe;
    }
    if (other.owner != self_owner && contact_precedes(probe, *contact)) {
        *contact = probe;
    }
}

fn scan_neighbours(
    center: vec3f,
    box: Aabb,
    radius: f32,
    self_index: u32,
    self_owner: u32,
    solid: ptr<function, ParticleContact>,
    contact: ptr<function, ParticleContact>,
    sensor: ptr<function, ParticleContact>,
) {
    let slices = grid_slices(box);
    for (var index = 0u; index < slices; index = index + 1u) {
        let slice = grid_slice(box, index);
        for (var entry = slice.first; entry < grid_scan_end(slice); entry = entry + 1u) {
            visit_entry(
                entry_node(entry),
                box,
                slice.cell_size,
                center,
                radius,
                self_index,
                self_owner,
                solid,
                contact,
                sensor,
            );
        }
    }
}

fn no_fact() -> SoftFact {
    var fact: SoftFact;
    fact.scene_target = NO_SLOT;
    fact.id = NO_BODY;
    fact.generation = 0u;
    return fact;
}

fn fact_of(contact: ParticleContact) -> SoftFact {
    var fact = no_fact();
    fact.scene_target = scene_target_of(contact.kind, contact.partner);
    fact.id = contact.id;
    fact.generation = contact.generation;
    return fact;
}

fn no_contact_record() -> SoftContact {
    var record: SoftContact;
    record.normal = vec3f(0.0, 1.0, 0.0);
    record.depth = 0.0;
    record.point = vec3f(0.0);
    record.friction = 0.0;
    record.partner = NO_SLOT;
    record.group = NO_BODY;
    record.kind = ENTRY_KIND_PARTICLE;
    record.partner_inverse_mass = 0.0;
    record.contact = no_fact();
    record.sensor = no_fact();
    return record;
}

fn work(index: u32) {
    let particle = particles[index];
    if (particle.owner == NO_BODY || bodies[particle.owner].sleeping != 0u) {
        contacts[index] = no_contact_record();
        return;
    }
    let radius = particle.position.w;
    let center = particle.position.xyz;
    var solid = no_contact();
    var contact = no_contact();
    var sensor = no_contact();
    if (radius > 0.0) {
        let reach = max(bitcast<f32>(counter_load(COUNTER_PARTICLE_REACH)), 0.0);
        scan_neighbours(
            center,
            reach_box(center, radius + reach),
            radius,
            index,
            particle.owner,
            &solid,
            &contact,
            &sensor,
        );
    }
    var record = no_contact_record();
    if (contact_touches(solid)) {
        record.normal = solid.normal;
        record.depth = -solid.separation;
        record.point = solid.point;
        record.friction = max(solid.friction, 0.0);
        record.partner = solid.partner;
        record.group = solid.group;
        record.kind = solid.kind;
        if (solid.kind == ENTRY_KIND_PARTICLE) {
            record.partner_inverse_mass = particles[solid.partner].prev_position.w;
        }
    }
    if (contact_touches(contact)) {
        record.contact = fact_of(contact);
    }
    if (contact_touches(sensor)) {
        record.sensor = fact_of(sensor);
    }
    contacts[index] = record;
}
