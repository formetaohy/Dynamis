@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read_write> particles: array<SoftParticle>;
@group(0) @binding(6) var<storage, read> pressure: array<vec4f>;
@group(0) @binding(7) var<storage, read> bodies: array<SoftBody>;

const SHIFT_SCALE: f32 = 65536.0;

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

fn visit_neighbours(
    center: vec3f,
    box: Aabb,
    lambda: f32,
    rest: f32,
    h2: f32,
    h9: f32,
    self_index: u32,
    shift: ptr<function, vec3i>,
) {
    let slices = grid_slices(box);
    for (var index = 0u; index < slices; index = index + 1u) {
        let slice = grid_slice(box, index);
        if (slice.whole) {
            counter_add(COUNTER_COARSE_NEIGHBOURS, 1u);
        }
        for (var entry = slice.first; entry < grid_scan_end(slice); entry = entry + 1u) {
            let node = entry_node(entry);
            if (reach_holds(node, box, slice.cell_size)) {
                visit(node, center, lambda, rest, h2, h9, self_index, shift);
            }
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
        visit_neighbours(
            center,
            reach_box(center, support),
            lambda,
            rest,
            h2,
            h9,
            index,
            &shift,
        );
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
