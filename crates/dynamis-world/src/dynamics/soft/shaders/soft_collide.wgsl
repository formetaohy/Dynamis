@group(0) @binding(3) var<uniform> params: StepParams;
@group(0) @binding(4) var<storage, read_write> particles: array<SoftParticle>;
@group(0) @binding(5) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(6) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(7) var<storage, read> colliders: array<Collider>;
@group(0) @binding(8) var<storage, read> collider_owners: array<u32>;
@group(0) @binding(9) var<storage, read> collider_aabbs: array<Aabb>;
@group(0) @binding(10) var<storage, read_write> reactions: array<atomic<u32>>;

const CELL_SCAN_BUDGET: u32 = 4096u;
const REACTION_SCALE: f32 = 65536.0;
const REACTION_WORDS: u32 = 8u;

struct ParticleContact {
    separation: f32,
    normal: vec3f,
    point: vec3f,
}

fn no_contact() -> ParticleContact {
    var contact: ParticleContact;
    contact.separation = 3.402823466e38;
    contact.normal = vec3f(0.0, 1.0, 0.0);
    contact.point = vec3f(0.0);
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

fn contact_precedes(candidate: ParticleContact, candidate_slot: u32, held: ParticleContact, held_slot: u32) -> bool {
    if (candidate.separation < held.separation) {
        return true;
    }
    return candidate.separation == held.separation && candidate_slot < held_slot;
}

fn visit_entry(
    slot: u32,
    box: Aabb,
    center: vec3f,
    radius: f32,
    held: ptr<function, ParticleContact>,
    held_slot: ptr<function, u32>,
) {
    if (!aabb_overlaps(collider_aabbs[slot], box)) {
        return;
    }
    let owner = collider_owners[slot];
    if (owner == NO_BODY) {
        return;
    }
    let collider = colliders[slot];
    if (collider.kind == SHAPE_NONE || (collider.flags & COLLIDER_SENSOR) != 0u) {
        return;
    }
    let contact = particle_contact(world_collider(body_states[owner], collider), center, radius);
    if (contact_precedes(contact, slot, *held, *held_slot)) {
        *held = contact;
        *held_slot = slot;
    }
}

fn scan_cell(
    level: u32,
    coord: vec3i,
    box: Aabb,
    center: vec3f,
    radius: f32,
    held: ptr<function, ParticleContact>,
    held_slot: ptr<function, u32>,
) {
    let range = entry_bounds_of(level, coord);
    for (var entry = range.x; entry < range.y; entry = entry + 1u) {
        visit_entry(entry_colliders[entry], box, center, radius, held, held_slot);
    }
}

fn scan_level(
    level: u32,
    box: Aabb,
    center: vec3f,
    radius: f32,
    held: ptr<function, ParticleContact>,
    held_slot: ptr<function, u32>,
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
        visit_entry(entry_colliders[entry], box, center, radius, held, held_slot);
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

fn commit(index: u32, particle: SoftParticle) {
    var updated = particle;
    updated.velocity = vec4f(
        (particle.position.xyz - particle.prev_position.xyz) / params.dt,
        particle.velocity.w,
    );
    particles[index] = updated;
}

fn work(index: u32) {
    var particle = particles[index];
    if (particle.owner == NO_BODY) {
        return;
    }
    let weight = particle.prev_position.w;
    let radius = particle.position.w;
    let center = particle.position.xyz;
    var held = no_contact();
    var held_slot = NO_SLOT;
    if (radius > 0.0) {
        var box: Aabb;
        box.min = center - vec3f(radius);
        box.max = center + vec3f(radius);
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
                            scan_cell(level, vec3i(x, y, z), box, center, radius, &held, &held_slot);
                        }
                    }
                }
            } else {
                scan_level(level, box, center, radius, &held, &held_slot);
            }
        }
    }
    if (held_slot == NO_SLOT || held.separation >= 0.0) {
        commit(index, particle);
        return;
    }
    let depth = -held.separation;
    let normal = held.normal;
    let owner = collider_owners[held_slot];
    let body = load_body(owner);
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
                owner,
                -normal * (lambda * body_weight),
                -apply_inverse_inertia(body, rn) * lambda,
            );
        }
    }
    if (weight > 0.0) {
        particle.position = vec4f(shifted, radius);
        let velocity = (particle.position.xyz - particle.prev_position.xyz) / params.dt;
        let tangential = velocity - normal * dot(velocity, normal);
        let slide = length(tangential) * params.dt;
        if (slide > 1e-6) {
            let collider = colliders[held_slot];
            let friction = sqrt(max(particle.velocity.w, 0.0) * collider.friction);
            let magnitude = min(slide, friction * depth);
            particle.position = vec4f(
                particle.position.xyz - normalize(tangential) * magnitude,
                radius,
            );
        }
    }
    commit(index, particle);
}

fn extent() -> u32 {
    return arrayLength(&particles);
}
