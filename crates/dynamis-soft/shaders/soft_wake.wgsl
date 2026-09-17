@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read> particles: array<SoftParticle>;
@group(0) @binding(6) var<storage, read_write> bodies: array<SoftBody>;
@group(0) @binding(7) var<storage, read> body_states: array<BodyState>;
@group(0) @binding(8) var<storage, read> body_descs: array<BodyDescriptor>;
@group(0) @binding(9) var<storage, read> colliders: array<Collider>;

fn load_body(row: u32) -> Body {
    return Body(body_states[row], body_descs[row]);
}

fn collider_pushes(body: Body) -> bool {
    if (body.state.sleeping != 0u || body_is_static(body)) {
        return false;
    }
    if (body_is_kinematic(body)) {
        return body_is_moving(body.state, body.desc, params)
            || any(body.state.position != body.state.prev_position);
    }
    return true;
}

fn partner_pushes(node: u32, owner: u32) -> bool {
    let info = entry_info(node);
    if (entry_kind(info) == ENTRY_KIND_COLLIDER) {
        let collider = colliders[entry_index(info)];
        if (collider.kind == SHAPE_NONE || (collider.flags & COLLIDER_SENSOR) != 0u) {
            return false;
        }
        let body_slot = entry_group(node);
        if (!filters_intersect(collider_filter(load_body(body_slot), collider), owner_filter(owner))) {
            return false;
        }
        return collider_pushes(load_body(body_slot));
    }
    let other = particles[entry_index(info)];
    if (other.owner == NO_BODY || !filters_intersect(owner_filter(other.owner), owner_filter(owner))) {
        return false;
    }
    return bodies[other.owner].sleeping == 0u;
}

fn visit_entry(node: u32, box: Aabb, cell_size: f32, owner: u32, held: ptr<function, bool>) {
    if (*held || !aabb_overlaps(entry_box(node), box)) {
        return;
    }
    if (!reach_holds(node, box, cell_size)) {
        return;
    }
    *held = partner_pushes(node, owner);
}

fn scan_neighbours(box: Aabb, owner: u32, held: ptr<function, bool>) {
    let view = entry_view();
    let slices = grid_slices(box);
    for (var index = 0u; index < slices && !(*held); index = index + 1u) {
        let slice = grid_slice(box, index);
        for (var entry = slice.first; entry < grid_scan_end(slice) && !(*held); entry = entry + 1u) {
            visit_entry(entry_node(view, entry), box, slice.cell_size, owner, held);
        }
    }
}

fn work(index: u32) {
    let particle = particles[index];
    if (particle.owner == NO_BODY || bodies[particle.owner].sleeping == 0u) {
        return;
    }
    let radius = particle.position.w;
    if (radius <= 0.0) {
        return;
    }
    let reach = max(bitcast<f32>(counter_load(COUNTER_PARTICLE_REACH)), 0.0);
    var held = false;
    scan_neighbours(reach_box(particle.position.xyz, radius + reach), particle.owner, &held);
    if (held) {
        atomicStore(&bodies[particle.owner].wake, 1u);
    }
}
