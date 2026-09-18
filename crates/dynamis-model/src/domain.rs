pub fn finite(value: f32, what: &str) -> f32 {
    assert!(value.is_finite(), "{what} must be finite");
    value
}

pub fn finite_vector(value: [f32; 3], what: &str) -> [f32; 3] {
    assert!(
        value.iter().all(|axis| axis.is_finite()),
        "{what} must be finite"
    );
    value
}

pub fn non_negative(value: f32, what: &str) -> f32 {
    assert!(
        value >= 0.0 && value.is_finite(),
        "{what} must be finite and non-negative"
    );
    value
}

pub fn positive(value: f32, what: &str) -> f32 {
    assert!(
        value > 0.0 && value.is_finite(),
        "{what} must be finite and strictly positive"
    );
    value
}

pub fn unit_quaternion(value: [f32; 4], what: &str) -> [f32; 4] {
    let square = value.iter().map(|axis| axis * axis).sum::<f32>();
    assert!(
        (square - 1.0).abs() < 1e-4,
        "{what} must be a unit quaternion"
    );
    value
}
