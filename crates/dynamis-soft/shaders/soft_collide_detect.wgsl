@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read> particles: array<SoftParticle>;
@group(0) @binding(6) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(7) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(8) var<storage, read> colliders: array<Collider>;
@group(0) @binding(9) var<storage, read> elements: array<SoftElement>;
@group(0) @binding(10) var<storage, read> adjacency: array<u32>;
@group(0) @binding(11) var<storage, read_write> contacts: array<SoftContact>;
@group(0) @binding(12) var<storage, read> bodies: array<SoftBody>;

struct ParticleContact {
    separation: f32,
    normal: vec3f,
    point: vec3f,
    kind: u32,
    partner: u32,
    group: u32,
    friction: f32,
}

fn no_contact() -> ParticleContact {
    var contact: ParticleContact;
    contact.separation = 3.402823466e38;
    contact.normal = vec3f(0.0, 1.0, 0.0);
    contact.point = vec3f(0.0);
    contact.kind = ENTRY_KIND_COLLIDER;
    contact.partner = NO_SLOT;
    contact.group = NO_BODY;
    contact.friction = 0.0;
    return contact;
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

fn visit_entry(
    node: u32,
    box: Aabb,
    cell_size: f32,
    center: vec3f,
    radius: f32,
    self_index: u32,
    self_owner: u32,
    held: ptr<function, ParticleContact>,
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
        if (collider.kind == SHAPE_NONE || (collider.flags & COLLIDER_SENSOR) != 0u) {
            return;
        }
        var triangle = NO_TRIANGLE;
        var contact = particle_contact(world_collider(body_states[entry_group(node)], collider), center, radius, &triangle);
        contact.kind = ENTRY_KIND_COLLIDER;
        contact.partner = entry_index(info);
        contact.group = entry_group(node);
        contact.friction = surface_material(collider, triangle).friction;
        if (contact_precedes(contact, *held)) {
            *held = contact;
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
    let delta = other.position.xyz - center;
    let distance = length(delta);
    let separation = distance - (radius + other.position.w);
    if (separation >= 0.0) {
        return;
    }
    var contact = no_contact();
    contact.separation = separation;
    contact.normal = sign_normalize(delta);
    contact.point = center + contact.normal * (radius - separation * 0.5);
    contact.kind = ENTRY_KIND_PARTICLE;
    contact.partner = other_index;
    contact.group = other.owner;
    contact.friction = other.velocity.w;
    if (contact_precedes(contact, *held)) {
        *held = contact;
    }
}

fn scan_neighbours(
    center: vec3f,
    box: Aabb,
    radius: f32,
    self_index: u32,
    self_owner: u32,
    held: ptr<function, ParticleContact>,
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
                held,
            );
        }
    }
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
    var held = no_contact();
    if (radius > 0.0) {
        let reach = max(bitcast<f32>(counter_load(COUNTER_PARTICLE_REACH)), 0.0);
        scan_neighbours(
            center,
            reach_box(center, radius + reach),
            radius,
            index,
            particle.owner,
            &held,
        );
    }
    var record = no_contact_record();
    if (held.partner != NO_SLOT && held.separation < 0.0) {
        record.normal = held.normal;
        record.depth = -held.separation;
        record.point = held.point;
        record.friction = max(held.friction, 0.0);
        record.partner = held.partner;
        record.group = held.group;
        record.kind = held.kind;
        if (held.kind == ENTRY_KIND_PARTICLE) {
            record.partner_inverse_mass = particles[held.partner].prev_position.w;
        }
    }
    contacts[index] = record;
}
