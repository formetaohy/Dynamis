struct ElementSample {
    valid: bool,
    value: f32,
    gradients: array<vec3f, ELEMENT_PARTICLES>,
}

fn element_arity(kind: u32) -> u32 {
    if (kind == ELEMENT_DISTANCE) {
        return 2u;
    }
    if (kind == ELEMENT_AREA) {
        return 3u;
    }
    return ELEMENT_PARTICLES;
}

fn element_strain(kind: u32, rest: f32, deviation: f32) -> f32 {
    if (kind == ELEMENT_BEND) {
        return abs(deviation);
    }
    return abs(deviation) / rest;
}

fn sample_unset() -> ElementSample {
    var sample: ElementSample;
    sample.valid = false;
    sample.value = 0.0;
    for (var slot = 0u; slot < ELEMENT_PARTICLES; slot = slot + 1u) {
        sample.gradients[slot] = vec3f(0.0);
    }
    return sample;
}

fn sample_distance(rest: f32, first: vec3f, second: vec3f) -> ElementSample {
    let offset = second - first;
    let span = length(offset);
    if (span < 1e-8) {
        return sample_unset();
    }
    let direction = offset / span;
    var sample = sample_unset();
    sample.valid = true;
    sample.value = span - rest;
    sample.gradients[0] = -direction;
    sample.gradients[1] = direction;
    return sample;
}

fn sample_area(rest: f32, first: vec3f, second: vec3f, third: vec3f) -> ElementSample {
    let doubled = cross(second - first, third - first);
    let extent = length(doubled);
    if (extent < 1e-12) {
        return sample_unset();
    }
    let normal = doubled / extent;
    var sample = sample_unset();
    sample.valid = true;
    sample.value = 0.5 * extent - rest;
    sample.gradients[1] = 0.5 * cross(third - first, normal);
    sample.gradients[2] = 0.5 * cross(normal, second - first);
    sample.gradients[0] = -(sample.gradients[1] + sample.gradients[2]);
    return sample;
}

fn sample_volume(rest: f32, first: vec3f, second: vec3f, third: vec3f, fourth: vec3f) -> ElementSample {
    let first_edge = second - first;
    let second_edge = third - first;
    let third_edge = fourth - first;
    let volume = dot(cross(first_edge, second_edge), third_edge) / 6.0;
    let orientation = select(1.0, -1.0, volume < 0.0);
    var sample = sample_unset();
    sample.valid = true;
    sample.value = abs(volume) - rest;
    sample.gradients[1] = orientation * cross(second_edge, third_edge) / 6.0;
    sample.gradients[2] = orientation * cross(third_edge, first_edge) / 6.0;
    sample.gradients[3] = orientation * cross(first_edge, second_edge) / 6.0;
    sample.gradients[0] = -(sample.gradients[1] + sample.gradients[2] + sample.gradients[3]);
    return sample;
}

fn sample_bend(rest: f32, apex_a: vec3f, apex_b: vec3f, edge_a: vec3f, edge_b: vec3f) -> ElementSample {
    let hinge = edge_b - edge_a;
    let span = length(hinge);
    if (span < 1e-8) {
        return sample_unset();
    }
    let inverse_span = 1.0 / span;
    let folded_a = cross(edge_a - apex_a, edge_b - apex_a);
    let folded_b = cross(edge_b - apex_b, edge_a - apex_b);
    let area_a = dot(folded_a, folded_a);
    let area_b = dot(folded_b, folded_b);
    if (area_a < 1e-12 || area_b < 1e-12) {
        return sample_unset();
    }
    let scaled_a = folded_a / area_a;
    let scaled_b = folded_b / area_b;
    let normal_a = folded_a / sqrt(area_a);
    let normal_b = folded_b / sqrt(area_b);
    let orientation = select(
        1.0,
        -1.0,
        dot(cross(normal_a, normal_b), hinge) > 0.0,
    );
    var sample = sample_unset();
    sample.valid = true;
    sample.value = acos(clamp(dot(normal_a, normal_b), -1.0, 1.0)) - rest;
    sample.gradients[0] = orientation * span * scaled_a;
    sample.gradients[1] = orientation * span * scaled_b;
    sample.gradients[2] = orientation
        * inverse_span
        * (dot(apex_a - edge_b, hinge) * scaled_a + dot(apex_b - edge_b, hinge) * scaled_b);
    sample.gradients[3] = orientation
        * inverse_span
        * (dot(edge_a - apex_a, hinge) * scaled_a + dot(edge_a - apex_b, hinge) * scaled_b);
    return sample;
}

fn element_sample(kind: u32, rest: f32, positions: array<vec3f, ELEMENT_PARTICLES>) -> ElementSample {
    switch (kind) {
        case ELEMENT_AREA: {
            return sample_area(rest, positions[0], positions[1], positions[2]);
        }
        case ELEMENT_BEND: {
            return sample_bend(rest, positions[0], positions[1], positions[2], positions[3]);
        }
        case ELEMENT_VOLUME: {
            return sample_volume(rest, positions[0], positions[1], positions[2], positions[3]);
        }
        default: {
            return sample_distance(rest, positions[0], positions[1]);
        }
    }
}
