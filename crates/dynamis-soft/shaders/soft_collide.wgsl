@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read_write> particles: array<SoftParticle>;
@group(0) @binding(6) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(7) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(8) var<storage, read> colliders: array<Collider>;
@group(0) @binding(9) var<storage, read> elements: array<SoftElement>;
@group(0) @binding(10) var<storage, read> adjacency: array<u32>;
@group(0) @binding(11) var<storage, read_write> reactions: array<atomic<u32>>;

const CELL_SCAN_BUDGET: u32 = 4096u;
const REACTION_SCALE: f32 = 65536.0;
const REACTION_WORDS: u32 = 8u;

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

fn load_body(row: u32) -> Body {
    return Body(body_states[row], body_descs[row]);
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

fn particle_contact(world: WorldShape, center: vec3f, radius: f32) -> ParticleContact {
    var contact = no_contact();
    if (world.kind == SHAPE_PLANE) {
        let normal = plane_normal(world);
        contact.separation = dot(center - world.center, normal) - radius;
        contact.normal = normal;
        contact.point = center - normal * radius;
        return contact;
    }
    if (world.kind == SHAPE_MESH || world.kind == SHAPE_HEIGHTFIELD) {
        var triangle = 0u;
        let closest = scene_convex_closest(world, sphere_probe(center, radius), &triangle);
        contact.separation = closest.distance;
        contact.normal = closest.normal;
        contact.point = closest.point_a;
        return contact;
    }
    let probe = sphere_probe(center, 0.0);
    var simplex: array<SimplexPoint, 4>;
    var count = 0u;
    let closest = convex_closest(probe, world, &simplex, &count);
    if (closest.penetrating) {
        let hit = convex_hit(probe, world);
        contact.separation = hit.distance - radius;
        contact.normal = -hit.normal;
        contact.point = hit.point;
        return contact;
    }
    contact.separation = closest.distance - radius;
    contact.normal = -closest.normal;
    contact.point = closest.point_b;
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
        let element = elements[adjacency[particle.neighbour_offset + slot] >> 1u];
        if (element.particles[0] == second || element.particles[1] == second) {
            return true;
        }
    }
    return false;
}

fn visit_entry(
    node: u32,
    box: Aabb,
    center: vec3f,
    radius: f32,
    self_index: u32,
    self_owner: u32,
    held: ptr<function, ParticleContact>,
) {
    if (!aabb_overlaps(entry_box(node), box)) {
        return;
    }
    let info = entries[node].info;
    if (entry_kind(info) == ENTRY_KIND_COLLIDER) {
        let collider = colliders[entry_index(info)];
        if (collider.kind == SHAPE_NONE || (collider.flags & COLLIDER_SENSOR) != 0u) {
            return;
        }
        var contact = particle_contact(world_collider(body_states[entry_group(node)], collider), center, radius);
        contact.kind = ENTRY_KIND_COLLIDER;
        contact.partner = entry_index(info);
        contact.group = entry_group(node);
        contact.friction = collider.friction;
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

fn scan_cell(
    level: u32,
    coord: vec3i,
    box: Aabb,
    center: vec3f,
    radius: f32,
    self_index: u32,
    self_owner: u32,
    held: ptr<function, ParticleContact>,
) {
    let range = entry_bounds_of(level, coord);
    for (var entry = range.x; entry < range.y; entry = entry + 1u) {
        visit_entry(entry_node(entry), box, center, radius, self_index, self_owner, held);
    }
}

fn scan_level(
    level: u32,
    box: Aabb,
    center: vec3f,
    radius: f32,
    self_index: u32,
    self_owner: u32,
    held: ptr<function, ParticleContact>,
) {
    let live = entry_live();
    let first = entry_bounds(live, level << LEVEL_KEY_SHIFT).x;
    let end = entry_bounds(live, (level + 1u) << LEVEL_KEY_SHIFT).x;
    var scanned = 0u;
    for (var entry = first; entry < end; entry = entry + 1u) {
        if (scanned >= CELL_SCAN_BUDGET) {
            break;
        }
        scanned = scanned + 1u;
        visit_entry(entry_node(entry), box, center, radius, self_index, self_owner, held);
    }
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
    if (index >= params.particle_count) {
        return;
    }
    var particle = particles[index];
    if (particle.owner == NO_BODY) {
        return;
    }
    let weight = particle.prev_position.w;
    let radius = particle.position.w;
    let center = particle.position.xyz;
    var held = no_contact();
    if (radius > 0.0) {
        let reach = max(bitcast<f32>(counter_load(COUNTER_PARTICLE_REACH)), 0.0);
        let half = radius + reach;
        var box: Aabb;
        box.min = center - vec3f(half);
        box.max = center + vec3f(half);
        let base_cell = grid_base_cell();
        var occupied = counter_load(COUNTER_GRID_LEVELS);
        while (occupied != 0u) {
            let level = countTrailingZeros(occupied);
            occupied = occupied & (occupied - 1u);
            let cell_size = level_cell_size(level, base_cell);
            let min_cell = vec3i(floor(box.min / cell_size));
            let max_cell = vec3i(floor(box.max / cell_size));
            let span = max_cell - min_cell + vec3i(1);
            let cells = u32(span.x) * u32(span.y) * u32(span.z);
            if (cells <= CELL_SCAN_BUDGET) {
                for (var x = min_cell.x; x <= max_cell.x; x = x + 1) {
                    for (var y = min_cell.y; y <= max_cell.y; y = y + 1) {
                        for (var z = min_cell.z; z <= max_cell.z; z = z + 1) {
                            scan_cell(
                                level,
                                vec3i(x, y, z),
                                box,
                                center,
                                radius,
                                index,
                                particle.owner,
                                &held,
                            );
                        }
                    }
                }
            } else {
                scan_level(
                    level,
                    box,
                    center,
                    radius,
                    index,
                    particle.owner,
                    &held,
                );
            }
        }
    }
    if (held.partner == NO_SLOT || held.separation >= 0.0) {
        particles[index] = particle;
        return;
    }
    let depth = -held.separation;
    let normal = held.normal;
    let friction = max(particle.velocity.w, 0.0) * max(held.friction, 0.0);
    if (held.kind == ENTRY_KIND_COLLIDER) {
        let body = load_body(held.group);
        var shifted = center + normal * depth;
        if (body.desc.inverse_mass > 0.0) {
            let ra = held.point - body_com(body);
            let rn = cross(ra, normal);
            let body_weight = body.desc.inverse_mass;
            let k = weight + body_weight + dot(rn, apply_inverse_inertia(body, rn));
            if (k > 0.0) {
                let lambda = depth / k;
                shifted = center + normal * (lambda * weight);
                accumulate_reaction(
                    held.group,
                    -normal * (lambda * body_weight),
                    -apply_inverse_inertia(body, rn) * lambda,
                );
            }
        }
        if (weight > 0.0) {
            particle.position = vec4f(shifted, radius);
            hold_contact_velocity(&particle, normal);
            apply_friction(&particle, normal, depth, friction);
        }
    } else if (weight > 0.0) {
        let other_weight = particles[held.partner].prev_position.w;
        let total = weight + other_weight;
        if (total > 0.0) {
            particle.position = vec4f(center - normal * (depth * weight / total), radius);
            hold_contact_velocity(&particle, normal);
            apply_friction(&particle, normal, depth, friction);
        }
    }
    particles[index] = particle;
}

fn extent() -> u32 {
    return arrayLength(&particles);
}
