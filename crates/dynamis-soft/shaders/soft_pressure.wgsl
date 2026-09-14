@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read_write> particles: array<SoftParticle>;
@group(0) @binding(6) var<storage, read> pressure: array<vec4f>;
@group(0) @binding(7) var<storage, read> bodies: array<SoftBody>;

const POLY6_SLOPE: f32 = 945.0 / (32.0 * 3.141592653589793);
const SHIFT_SCALE: f32 = 65536.0;

fn fixed(value: f32, scale: f32) -> i32 {
    return i32(round(value * scale));
}

fn poly6_slope(r: f32, r2: f32, h2: f32, h9: f32) -> f32 {
    let span = h2 - r2;
    return -POLY6_SLOPE * r * span * span / h9;
}

fn visit(
    node: u32,
    center: vec3f,
    lambda: f32,
    rest: f32,
    h2: f32,
    h9: f32,
    self_index: u32,
    shift: ptr<function, vec3i>,
) {
    let info = entry_info(node);
    if (entry_kind(info) != ENTRY_KIND_PARTICLE) {
        return;
    }
    let other_index = entry_index(info);
    if (other_index == self_index) {
        return;
    }
    let other = particles[other_index];
    if (other.owner == NO_BODY || other.support <= 0.0 || other.prev_position.w <= 0.0) {
        return;
    }
    let offset = other.position.xyz - center;
    let r2 = dot(offset, offset);
    if (r2 >= h2 || r2 <= 0.0) {
        return;
    }
    let r = sqrt(r2);
    let neighbour = pressure[other_index].x;
    let magnitude = (lambda + neighbour) * poly6_slope(r, r2, h2, h9) / (rest * r);
    let direction = (center - other.position.xyz) * magnitude;
    (*shift) = (*shift)
        + vec3i(
            fixed(direction.x, SHIFT_SCALE),
            fixed(direction.y, SHIFT_SCALE),
            fixed(direction.z, SHIFT_SCALE),
        );
}

fn visit_level(
    level: u32,
    cell_size: f32,
    box: Aabb,
    center: vec3f,
    lambda: f32,
    rest: f32,
    h2: f32,
    h9: f32,
    self_index: u32,
    shift: ptr<function, vec3i>,
) {
    let live = entry_live();
    let first = entry_bounds(live, level << LEVEL_KEY_SHIFT).x;
    let end = entry_bounds(live, (level + 1u) << LEVEL_KEY_SHIFT).x;
    counter_add(COUNTER_COARSE_NEIGHBOURS, 1u);
    var scanned = 0u;
    for (var entry = first; entry < end; entry = entry + 1u) {
        if (scanned >= REACH_CELL_BUDGET) {
            break;
        }
        scanned = scanned + 1u;
        let node = entry_node(entry);
        if (reach_holds(node, box, cell_size)) {
            visit(node, center, lambda, rest, h2, h9, self_index, shift);
        }
    }
}

fn work(index: u32) {
    var particle = particles[index];
    let support = particle.support;
    if (particle.owner == NO_BODY
        || support <= 0.0
        || particle.prev_position.w <= 0.0
        || bodies[particle.owner].sleeping != 0u) {
        return;
    }
    let lambda = pressure[index].x;
    let rest = pressure[index].y;
    let radius = particle.position.w;
    let center = particle.position.xyz;
    var shift = vec3i(0);
    if (lambda < 0.0 && rest > 0.0) {
        let h2 = support * support;
        let h3 = support * support * support;
        let h9 = h3 * h3 * h3;
        let box = reach_box(center, support);
        let base_cell = grid_base_cell();
        var occupied = counter_load(COUNTER_GRID_LEVELS);
        while (occupied != 0u) {
            let level = countTrailingZeros(occupied);
            occupied = occupied & (occupied - 1u);
            let cell_size = level_cell_size(level, base_cell);
            let cells = reach_cells(box, cell_size);
            if (reach_span(cells) <= REACH_CELL_BUDGET) {
                for (var x = cells.min.x; x <= cells.max.x; x = x + 1) {
                    for (var y = cells.min.y; y <= cells.max.y; y = y + 1) {
                        for (var z = cells.min.z; z <= cells.max.z; z = z + 1) {
                            let range = entry_bounds_of(level, vec3i(x, y, z));
                            for (var entry = range.x; entry < range.y; entry = entry + 1u) {
                                let node = entry_node(entry);
                                if (reach_holds(node, box, cell_size)) {
                                    visit(node, center, lambda, rest, h2, h9, index, &shift);
                                }
                            }
                        }
                    }
                }
            } else {
                visit_level(level, cell_size, box, center, lambda, rest, h2, h9, index, &shift);
            }
        }
    }
    var offset = vec3f(shift) / SHIFT_SCALE;
    let reach = length(offset);
    let limit = 0.25 * radius;
    if (reach > limit) {
        offset = offset * (limit / reach);
    }
    particle.position = vec4f(center + offset, radius);
    particles[index] = particle;
}
