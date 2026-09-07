use dynamis_model::{BodyDesc, ColliderDesc, Shape, compute_mass_properties};

fn inverse_mass(inv: [f32; 6], axis: usize) -> f32 {
    let xx = inv[0];
    let yy = inv[3];
    let zz = inv[5];
    let index = axis % 3;
    [xx, yy, zz][index]
}

#[test]
fn analytic_sphere_inertia_matches_closed_form() {
    let mass = BodyDesc::sphere(0.5).mass(2.0).mass_properties(|_| None);
    let expected_inv = 1.0 / (0.4 * 2.0 * 0.25);
    assert_eq!(mass.com, [0.0; 3]);
    assert!(mass.inverse_inertia[0] == mass.inverse_inertia[3]);
    assert!(mass.inverse_inertia[0] == mass.inverse_inertia[5]);
    assert!((mass.inverse_inertia[0] - expected_inv).abs() < 1e-6);
    assert_eq!(mass.inverse_inertia[1], 0.0);
    assert_eq!(mass.inverse_inertia[2], 0.0);
    assert_eq!(mass.inverse_inertia[4], 0.0);
}

#[test]
fn static_and_sensor_only_bodies_carry_no_inertia() {
    let static_body = BodyDesc::static_sphere(0.5).mass_properties(|_| None);
    assert_eq!(static_body.com, [0.0; 3]);
    assert_eq!(static_body.inverse_inertia, [0.0; 6]);

    let sensor_only =
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).sensor(true)).mass_properties(|_| None);
    assert_eq!(sensor_only.com, [0.0; 3]);
    assert_eq!(sensor_only.inverse_inertia, [0.0; 6]);
}

#[test]
fn dumbell_parallel_axis_matches_closed_form() {
    let radius = 0.1f32;
    let separation = 2.0f32;
    let mass = 2.0f32;
    let desc = BodyDesc::new(ColliderDesc::new(Shape::sphere(radius)))
        .collider(ColliderDesc::new(Shape::sphere(radius)).offset([separation, 0.0, 0.0]))
        .mass(mass);
    let properties = desc.mass_properties(|_| None);
    assert_eq!(properties.com, [1.0, 0.0, 0.0]);
    let sphere_i = 0.4 * (mass * 0.5) * radius * radius;
    let axis_i = sphere_i + (mass * 0.5) * (separation * 0.5) * (separation * 0.5);
    let expected_inv_z = 1.0 / (2.0 * axis_i);
    let expected_inv_x = 1.0 / (2.0 * sphere_i);
    assert!((inverse_mass(properties.inverse_inertia, 0) - expected_inv_x).abs() < 1e-4);
    assert!((inverse_mass(properties.inverse_inertia, 2) - expected_inv_z).abs() < 1e-4);
}

#[test]
fn com_override_relocates_inertia_axis() {
    let desc = BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).offset([2.0, 0.0, 0.0]))
        .mass(1.0)
        .com([2.0, 0.0, 0.0]);
    let properties = desc.mass_properties(|_| None);
    assert_eq!(properties.com, [2.0, 0.0, 0.0]);
    let expected_inv = 1.0 / (0.4 * 0.25);
    assert!((inverse_mass(properties.inverse_inertia, 0) - expected_inv).abs() < 1e-5);
}

#[test]
fn collider_rotation_swaps_box_inertia_axes() {
    let desc = BodyDesc::new(
        ColliderDesc::new(Shape::cuboid([1.0, 0.1, 0.1]))
            .rotation([0.0, 0.0, 0.7071068, 0.7071068]),
    )
    .mass(1.0);
    let properties = desc.mass_properties(|_| None);
    let ix = 1.0 / inverse_mass(properties.inverse_inertia, 0);
    let iy = 1.0 / inverse_mass(properties.inverse_inertia, 1);
    assert!((ix - 1.0 / 12.0 * (0.04 + 4.0)).abs() < 1e-4);
    assert!((iy - 1.0 / 12.0 * (0.04 + 0.04)).abs() < 1e-4);
}

#[test]
fn inertia_override_bypasses_composite() {
    let desc = BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).offset([3.0, 0.0, 0.0]))
        .mass(1.0)
        .com([3.0, 0.0, 0.0])
        .inertia([2.0, 0.0, 0.0, 2.0, 0.0, 2.0]);
    let properties = desc.mass_properties(|_| None);
    assert_eq!(properties.com, [3.0, 0.0, 0.0]);
    assert!((inverse_mass(properties.inverse_inertia, 0) - 0.5).abs() < 1e-6);
}

#[test]
fn world_geometry_uses_bounds_approximation() {
    let handle = dynamis_model::ShapeSourceHandle {
        id: 0,
        generation: 1,
    };
    let desc = BodyDesc::new(ColliderDesc::new(Shape::hull(handle))).mass(1.0);
    let properties = desc.mass_properties(|_| Some(([0.0; 3], [2.0, 2.0, 2.0])));
    let expected = 1.0 / (1.0 / 12.0 * (4.0 + 4.0));
    assert!((inverse_mass(properties.inverse_inertia, 0) - expected).abs() < 1e-5);
}

#[test]
fn composite_across_many_colliders_is_consistent() {
    let desc = BodyDesc::new(ColliderDesc::new(Shape::sphere(0.2)))
        .collider(ColliderDesc::new(Shape::sphere(0.2)).offset([0.0, 1.0, 0.0]))
        .collider(ColliderDesc::new(Shape::sphere(0.2)).offset([0.0, 2.0, 0.0]))
        .mass(3.0);
    let properties = desc.mass_properties(|_| None);
    assert_eq!(properties.com, [0.0, 1.0, 0.0]);
    let z_expected = 1.0 / (3.0 * 0.4 * 0.04 + 2.0 * 1.0);
    assert!((inverse_mass(properties.inverse_inertia, 2) - z_expected).abs() < 1e-4);
}

#[test]
fn degenerate_inertia_overrides_panic() {
    let desc = BodyDesc::sphere(0.5)
        .mass(1.0)
        .inertia([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let result = std::panic::catch_unwind(|| desc.mass_properties(|_| None));
    assert!(result.is_err());
}

#[test]
fn compute_mass_properties_distributes_mass_over_solids() {
    let colliders = [
        ColliderDesc::new(Shape::sphere(1.0)),
        ColliderDesc::new(Shape::sphere(1.0)).offset([0.0, 2.0, 1.0]),
    ];
    let properties = compute_mass_properties(&colliders, 4.0, None, |_| None);
    assert_eq!(properties.com, [0.0, 1.0, 0.5]);
    assert!(properties.inverse_inertia.iter().all(|value| *value >= 0.0));
    assert!(properties.inverse_inertia[0] > 0.0);
}
