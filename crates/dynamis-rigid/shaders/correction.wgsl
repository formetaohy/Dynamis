struct Correction {
    linear: vec3f,
    angular: vec3f,
}

struct CorrectionPair {
    first: Correction,
    second: Correction,
}

fn correction_zero() -> Correction {
    var correction: Correction;
    correction.linear = vec3f(0.0);
    correction.angular = vec3f(0.0);
    return correction;
}

fn correction_holds(correction: Correction) -> bool {
    return dot(correction.linear, correction.linear) > 0.0 || dot(correction.angular, correction.angular) > 0.0;
}

fn pair_zero() -> CorrectionPair {
    var pair: CorrectionPair;
    pair.first = correction_zero();
    pair.second = correction_zero();
    return pair;
}

fn pair_holds(pair: CorrectionPair) -> bool {
    return correction_holds(pair.first) || correction_holds(pair.second);
}

fn apply_correction(body: ptr<function, Body>, correction: Correction) {
    (*body).state.position = (*body).state.position + correction.linear;
    (*body).state.orientation = normalize(quat_mul(vec4f(correction.angular * 0.5, 1.0), (*body).state.orientation));
}

fn row(
    first: ptr<function, Body>,
    second: ptr<function, Body>,
    total: ptr<function, CorrectionPair>,
    correction: CorrectionPair,
) {
    apply_correction(first, correction.first);
    apply_correction(second, correction.second);
    (*total).first.linear = (*total).first.linear + correction.first.linear;
    (*total).first.angular = (*total).first.angular + correction.first.angular;
    (*total).second.linear = (*total).second.linear + correction.second.linear;
    (*total).second.angular = (*total).second.angular + correction.second.angular;
}

fn point_row(
    axis: vec3f,
    point_a: vec3f,
    point_b: vec3f,
    goal: f32,
    scale: f32,
    first: Body,
    second: Body,
) -> CorrectionPair {
    var pair = pair_zero();
    let k = linear_momentum_mass(first, second);
    if (k > 0.0) {
        let magnitude = scale * (dot(point_b - point_a, axis) - goal) / k;
        pair.first.linear = axis * (magnitude * first.desc.inverse_mass);
        pair.second.linear = axis * (-magnitude * second.desc.inverse_mass);
    }
    return pair;
}

fn angular_row(axis: vec3f, error: f32, scale: f32, first: Body, second: Body) -> CorrectionPair {
    var pair = pair_zero();
    let k = dot(axis, apply_inverse_inertia(first, axis) + apply_inverse_inertia(second, axis));
    if (k > 0.0) {
        let magnitude = scale * error / k;
        pair.first.angular = apply_inverse_inertia(first, axis * magnitude);
        pair.second.angular = apply_inverse_inertia(second, axis * -magnitude);
    }
    return pair;
}

fn local_row(local_axis: vec3f, error: f32, scale: f32, first: Body, second: Body) -> CorrectionPair {
    return angular_row(quat_rotate(first.state.orientation, local_axis), error, scale, first, second);
}

fn local_point_row(
    local_a: vec3f,
    local_b: vec3f,
    axis: vec3f,
    goal: f32,
    scale: f32,
    first: Body,
    second: Body,
) -> CorrectionPair {
    return point_row(
        axis,
        constraint_anchor(first, local_a),
        constraint_anchor(second, local_b),
        goal,
        scale,
        first,
        second,
    );
}
