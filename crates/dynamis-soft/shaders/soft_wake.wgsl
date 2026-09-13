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

fn partner_pushes(node: u32) -> bool {
    let info = entry_info(node);
    if (entry_kind(info) == ENTRY_KIND_COLLIDER) {
        let collider = colliders[entry_index(info)];
        if (collider.kind == SHAPE_NONE || (collider.flags & COLLIDER_SENSOR) != 0u) {
            return false;
        }
        return collider_pushes(load_body(entry_group(node)));
    }
    let other = particles[entry_index(info)];
    return other.owner != NO_BODY && bodies[other.owner].sleeping == 0u;
}

fn visit_entry(node: u32, box: Aabb, cell_size: f32, held: ptr<function, bool>) {
    if (*held || !aabb_overlaps(entry_box(node), box)) {
        return;
    }
    if (!reach_holds(node, box, cell_size)) {
        return;
    }
    *held = partner_pushes(node);
}

fn scan_cell(level: u32, cell_size: f32, coord: vec3i, box: Aabb, held: ptr<function, bool>) {
    let range = entry_bounds_of(level, coord);
    for (var entry = range.x; entry < range.y; entry = entry + 1u) {
        visit_entry(entry_node(entry), box, cell_size, held);
    }
}

fn scan_level(level: u32, box: Aabb, held: ptr<function, bool>) {
    let live = entry_live();
    let first = entry_bounds(live, level << LEVEL_KEY_SHIFT).x;
    let end = entry_bounds(live, (level + 1u) << LEVEL_KEY_SHIFT).x;
    let cell_size = level_cell_size(level, grid_base_cell());
    var scanned = 0u;
    for (var entry = first; entry < end; entry = entry + 1u) {
        if (scanned >= REACH_CELL_BUDGET) {
            break;
        }
        scanned = scanned + 1u;
        visit_entry(entry_node(entry), box, cell_size, held);
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
    let box = reach_box(particle.position.xyz, radius + reach);
    var held = false;
    var occupied = counter_load(COUNTER_GRID_LEVELS);
    while (occupied != 0u && !held) {
        let level = countTrailingZeros(occupied);
        occupied = occupied & (occupied - 1u);
        let cell_size = level_cell_size(level, grid_base_cell());
        let cells = reach_cells(box, cell_size);
        if (reach_span(cells) <= REACH_CELL_BUDGET) {
            for (var x = cells.min.x; x <= cells.max.x; x = x + 1) {
                for (var y = cells.min.y; y <= cells.max.y; y = y + 1) {
                    for (var z = cells.min.z; z <= cells.max.z; z = z + 1) {
                        scan_cell(level, cell_size, vec3i(x, y, z), box, &held);
                    }
                }
            }
        } else {
            scan_level(level, box, &held);
        }
    }
    if (held) {
        atomicStore(&bodies[particle.owner].wake, 1u);
    }
}
