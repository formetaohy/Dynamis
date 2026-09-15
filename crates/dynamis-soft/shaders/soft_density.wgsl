@group(0) @binding(4) var<uniform> params: StepParams;
@group(0) @binding(5) var<storage, read> particles: array<SoftParticle>;
@group(0) @binding(6) var<storage, read_write> pressure: array<vec4f>;
@group(0) @binding(7) var<storage, read> bodies: array<SoftBody>;

const DENSITY_SCALE: f32 = 65536.0;
const SQUARE_SCALE: f32 = 64.0;

struct Sample {
    density: i32,
    slope: vec3i,
    square: i32,
}

fn rest_density(spacing: f32, h2: f32, h9: f32) -> f32 {
    var density = poly6(0.0, h2, h9);
    for (var x = -1; x <= 1; x = x + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var z = -1; z <= 1; z = z + 1) {
                if (x == 0 && y == 0 && z == 0) {
                    continue;
                }
                let r2 = spacing * spacing * f32(x * x + y * y + z * z);
                if (r2 < h2) {
                    density = density + poly6(r2, h2, h9);
                }
            }
        }
    }
    return density;
}

fn visit(
    node: u32,
    center: vec3f,
    h2: f32,
    h9: f32,
    self_index: u32,
    sample: ptr<function, Sample>,
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
    if (!filters_intersect(owner_filter(other.owner), owner_filter(particles[self_index].owner))) {
        return;
    }
    let offset = other.position.xyz - center;
    let r2 = dot(offset, offset);
    if (r2 >= h2 || r2 <= 0.0) {
        return;
    }
    let r = sqrt(r2);
    let slope = poly6_slope(r, r2, h2, h9);
    (*sample).density = (*sample).density + fixed(poly6(r2, h2, h9), DENSITY_SCALE);
    let direction = offset * (slope / r);
    (*sample).slope = (*sample).slope
        + vec3i(
            fixed(direction.x, DENSITY_SCALE),
            fixed(direction.y, DENSITY_SCALE),
            fixed(direction.z, DENSITY_SCALE),
        );
    (*sample).square = (*sample).square + fixed(slope * slope, SQUARE_SCALE);
}

fn visit_neighbours(
    center: vec3f,
    box: Aabb,
    h2: f32,
    h9: f32,
    self_index: u32,
    sample: ptr<function, Sample>,
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
                visit(node, center, h2, h9, self_index, sample);
            }
        }
    }
}

fn work(index: u32) {
    let particle = particles[index];
    let support = particle.support;
    if (particle.owner == NO_BODY
        || support <= 0.0
        || particle.prev_position.w <= 0.0
        || bodies[particle.owner].sleeping != 0u) {
        pressure[index] = vec4f(0.0);
        return;
    }
    let center = particle.position.xyz;
    let h2 = support * support;
    let h3 = support * support * support;
    let h9 = h3 * h3 * h3;
    var sample: Sample;
    sample.density = fixed(poly6(0.0, h2, h9), DENSITY_SCALE);
    sample.slope = vec3i(0);
    sample.square = 0;
    let box = reach_box(center, support);
    visit_neighbours(center, box, h2, h9, index, &sample);
    let rest = rest_density(particle.rest_spacing, h2, h9);
    let density = f32(sample.density) / DENSITY_SCALE;
    let compression = max(density / rest - 1.0, 0.0);
    let slope = vec3f(sample.slope) / DENSITY_SCALE;
    let square = f32(sample.square) / SQUARE_SCALE;
    let gradient = (dot(slope, slope) + square) / (rest * rest);
    let lambda = select(0.0, -compression / gradient, gradient > 0.0);
    pressure[index] = vec4f(lambda, rest, density, 0.0);
}
