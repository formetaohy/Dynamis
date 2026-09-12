use dynamis_math::{length, quat_conjugate, quat_mul, quat_rotate};

fn rotation(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let half = angle * 0.5;
    let axis = dynamis_math::normalize(axis);
    let sine = half.sin();
    [axis[0] * sine, axis[1] * sine, axis[2] * sine, half.cos()]
}

fn assert_close(first: [f32; 3], second: [f32; 3], tolerance: f32) {
    let error = length([
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    ]);
    assert!(
        error < tolerance,
        "{first:?} must match {second:?}, off by {error}"
    );
}

#[test]
fn rotation_preserves_length() {
    let q = rotation([1.0, 2.0, 3.0], 1.1);
    for v in [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.3, 0.7, -0.2],
        [0.0, 0.0, 0.0],
    ] {
        let rotated = quat_rotate(q, v);
        assert!(
            (length(rotated) - length(v)).abs() < 1e-6,
            "a rotation must preserve {v:?}, got {rotated:?}"
        );
    }
}

#[test]
fn conjugate_undoes_a_rotation() {
    let q = rotation([1.0, -2.0, 0.5], 2.4);
    let v = [0.3, 0.7, -0.2];
    assert_close(quat_rotate(quat_conjugate(q), quat_rotate(q, v)), v, 1e-6);
}

#[test]
fn rotation_composes_through_quat_mul() {
    let first = rotation([0.0, 1.0, 0.0], 0.7);
    let second = rotation([1.0, 0.0, 1.0], 1.3);
    let v = [0.4, -0.6, 0.2];
    assert_close(
        quat_rotate(quat_mul(first, second), v),
        quat_rotate(first, quat_rotate(second, v)),
        1e-6,
    );
}

#[test]
fn quarter_turns_land_on_the_expected_axes() {
    let q = rotation([0.0, 0.0, 1.0], std::f32::consts::FRAC_PI_2);
    assert_close(quat_rotate(q, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], 1e-6);
    assert_close(quat_rotate(q, [0.0, 1.0, 0.0]), [-1.0, 0.0, 0.0], 1e-6);
    assert_close(quat_rotate(q, [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], 1e-6);
}
