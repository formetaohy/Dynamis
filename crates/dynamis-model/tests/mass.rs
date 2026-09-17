use dynamis_model::{
    BodyDesc, ColliderDesc, MassProperties, MassSource, Shape, SolidGeometry,
    compute_mass_properties,
};

fn matrix_of(tensor: [f32; 6]) -> [[f32; 3]; 3] {
    [
        [tensor[0], tensor[1], tensor[2]],
        [tensor[1], tensor[3], tensor[4]],
        [tensor[2], tensor[4], tensor[5]],
    ]
}

fn assert_mutual_inverse(properties: &MassProperties) {
    let tensor = matrix_of(properties.inertia);
    let inverse = matrix_of(properties.inverse_inertia);
    for (row, entries) in tensor.iter().enumerate() {
        for (other, column) in inverse.iter().enumerate() {
            let expected = if row == other { 1.0 } else { 0.0 };
            let product = entries
                .iter()
                .zip(column)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            assert!(
                (product - expected).abs() < 1e-4,
                "inertia and its inverse must be reciprocal, entry ({row}, {other}) is {product}"
            );
        }
    }
}

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
    let expected = 0.4 * 2.0 * 0.25;
    assert!((mass.inertia[0] - expected).abs() < 1e-6);
    assert!(mass.inertia[0] == mass.inertia[3]);
    assert!(mass.inertia[3] == mass.inertia[5]);
    assert_mutual_inverse(&mass);
}

#[test]
fn static_and_sensor_only_bodies_carry_no_inertia() {
    let static_body = BodyDesc::static_sphere(0.5).mass_properties(|_| None);
    assert_eq!(static_body.com, [0.0; 3]);
    assert_eq!(static_body.inertia, [0.0; 6]);
    assert_eq!(static_body.inverse_inertia, [0.0; 6]);

    let sensor_only =
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).sensor(true)).mass_properties(|_| None);
    assert_eq!(sensor_only.com, [0.0; 3]);
    assert_eq!(sensor_only.inertia, [0.0; 6]);
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
    assert!((properties.inertia[0] - 2.0 * sphere_i).abs() < 1e-4);
    assert!((properties.inertia[5] - 2.0 * axis_i).abs() < 1e-4);
    assert_mutual_inverse(&properties);
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
    let desc = BodyDesc::new(ColliderDesc::new(Shape::cuboid([1.0, 0.1, 0.1])).rotation([
        0.0,
        0.0,
        0.5_f32.sqrt(),
        0.5_f32.sqrt(),
    ]))
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
    assert_eq!(properties.inertia, [2.0, 0.0, 0.0, 2.0, 0.0, 2.0]);
    assert!((inverse_mass(properties.inverse_inertia, 0) - 0.5).abs() < 1e-6);
}

#[test]
fn rotated_composites_keep_inertia_and_its_inverse_reciprocal() {
    let rotation = [0.1825742, 0.3651484, 0.5477226, 0.7302967];
    let desc = BodyDesc::new(ColliderDesc::new(Shape::cuboid([0.5, 0.25, 1.0])).rotation(rotation))
        .collider(
            ColliderDesc::new(Shape::sphere(0.4))
                .offset([0.75, 0.5, -0.25])
                .rotation(rotation),
        )
        .mass(2.5);
    let properties = desc.mass_properties(|_| None);
    assert!(
        properties.inertia[1].abs() > 1e-4,
        "a rotated compound must carry off diagonal inertia"
    );
    assert_mutual_inverse(&properties);
}

fn unit_tetrahedron() -> SolidGeometry {
    SolidGeometry {
        volume: 1.0 / 6.0,
        centroid: [0.25, 0.25, 0.25],
        unit_second_moment: [
            3.0 / 80.0,
            -1.0 / 80.0,
            -1.0 / 80.0,
            3.0 / 80.0,
            -1.0 / 80.0,
            3.0 / 80.0,
        ],
    }
}

#[test]
fn a_hull_takes_its_mass_properties_from_its_solid_geometry() {
    let handle = dynamis_model::ShapeSourceHandle {
        id: 0,
        generation: 1,
    };
    let desc = BodyDesc::new(ColliderDesc::new(Shape::hull(handle))).mass(1.0);
    let properties = desc.mass_properties(|_| Some(unit_tetrahedron()));
    assert_eq!(properties.com, [0.25, 0.25, 0.25]);
    assert!((properties.inertia[0] - 3.0 / 40.0).abs() < 1e-6);
    assert!((properties.inertia[1] - 1.0 / 80.0).abs() < 1e-6);
    assert_mutual_inverse(&properties);
}

#[test]
fn composite_hulls_center_on_their_geometry_and_carry_off_diagonal_inertia() {
    let handle = dynamis_model::ShapeSourceHandle {
        id: 0,
        generation: 1,
    };
    let desc = BodyDesc::new(ColliderDesc::new(Shape::hull(handle)))
        .collider(ColliderDesc::new(Shape::hull(handle)).offset([0.0, 0.0, 1.5]))
        .mass(2.0);
    let properties = desc.mass_properties(|_| Some(unit_tetrahedron()));
    assert!(
        (properties.com[2] - 1.0).abs() < 1e-5,
        "two equal hulls must center between their geometry centroids, got {}",
        properties.com[2]
    );
    assert!(
        properties.inertia[2].abs() > 1e-4,
        "an offset hull must carry off diagonal inertia"
    );
    assert_mutual_inverse(&properties);
}

#[test]
#[should_panic(expected = "must answer its solid geometry")]
fn a_hull_without_its_solid_geometry_is_refused() {
    let handle = dynamis_model::ShapeSourceHandle {
        id: 0,
        generation: 1,
    };
    BodyDesc::new(ColliderDesc::new(Shape::hull(handle)))
        .mass(1.0)
        .mass_properties(|_| None);
}

#[test]
fn surface_geometry_carries_no_mass() {
    let handle = dynamis_model::ShapeSourceHandle {
        id: 0,
        generation: 1,
    };
    for shape in [
        Shape::mesh(handle),
        Shape::height_field(handle),
        Shape::plane(),
    ] {
        let desc = BodyDesc::new(ColliderDesc::new(shape)).density(1.0);
        assert!(desc.effective_mass(|_| None) == 0.0);
        let properties = desc.mass_properties(|_| None);
        assert_eq!(properties.inertia, [0.0; 6]);
        assert_eq!(properties.com, [0.0; 3]);
    }
}

#[test]
fn capsule_inertia_matches_its_hemisphere_composite() {
    let radius = 0.5f32;
    let half_height = 1.0f32;
    let desc = BodyDesc::capsule(radius, half_height).mass(1.0);
    let properties = desc.mass_properties(|_| None);
    assert!(
        (properties.inertia[0] - 0.665625).abs() < 1e-5,
        "lateral capsule inertia must match the hemisphere composite, got {}",
        properties.inertia[0]
    );
    assert!(
        (properties.inertia[3] - 0.11875).abs() < 1e-5,
        "axial capsule inertia must match the hemisphere composite, got {}",
        properties.inertia[3]
    );
    assert_eq!(properties.inertia[0], properties.inertia[5]);
    assert_mutual_inverse(&properties);
}

#[test]
fn a_degenerate_capsule_reduces_to_a_sphere() {
    let desc = BodyDesc::capsule(1.0, 0.0).mass(1.0);
    let properties = desc.mass_properties(|_| None);
    let sphere = BodyDesc::sphere(1.0).mass(1.0).mass_properties(|_| None);
    for index in 0..6 {
        assert!(
            (properties.inertia[index] - sphere.inertia[index]).abs() < 1e-5,
            "a capsule without a cylinder must be its sphere, axis {index}"
        );
    }
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
fn an_inertia_that_is_not_positive_definite_is_refused() {
    let desc = BodyDesc::sphere(0.5)
        .mass(1.0)
        .inertia([-1.0, 0.0, 0.0, -1.0, 0.0, 1.0]);
    let result = std::panic::catch_unwind(|| desc.mass_properties(|_| None));
    assert!(result.is_err());
}

#[test]
fn a_slender_body_keeps_its_closed_form_inertia() {
    for ratio in [100.0f32, 1_000.0, 10_000.0, 1_000_000.0] {
        let desc = BodyDesc::cuboid([ratio, 1.0, 1.0]).mass(1.0);
        let properties = desc.mass_properties(|_| None);
        let lateral = (4.0 + 4.0) / 12.0;
        let axial = (4.0 * ratio * ratio + 4.0) / 12.0;
        assert!(
            (properties.inertia[0] - lateral).abs() < 1e-6,
            "a {ratio}:1 rod must keep I_xx = {lateral}, got {}",
            properties.inertia[0]
        );
        assert!(
            (properties.inertia[3] / axial - 1.0).abs() < 1e-6,
            "a {ratio}:1 rod must keep I_yy = {axial}, got {}",
            properties.inertia[3]
        );
        assert_eq!(properties.inertia[3], properties.inertia[5]);
        assert_mutual_inverse(&properties);
    }
}

#[test]
fn a_slender_cylinder_keeps_its_closed_form_inertia() {
    for half_height in [1.0f32, 10_000.0, 1_000_000.0] {
        let desc = BodyDesc::cylinder(1.0, half_height).mass(1.0);
        let properties = desc.mass_properties(|_| None);
        assert!(
            (properties.inertia[3] - 0.5).abs() < 1e-6,
            "a cylinder about its own axis must keep I_yy = r²/2, got {}",
            properties.inertia[3]
        );
        let lateral = (3.0 + 4.0 * half_height * half_height) / 12.0;
        assert!(
            (properties.inertia[0] / lateral - 1.0).abs() < 1e-6,
            "a {half_height}:1 cylinder must keep I_xx = {lateral}, got {}",
            properties.inertia[0]
        );
        assert_mutual_inverse(&properties);
    }
}

#[test]
fn a_scaled_collider_composes_its_second_moment_exactly() {
    let cube =
        BodyDesc::new(ColliderDesc::new(Shape::cuboid([0.5, 0.5, 0.5])).scale([1_000.0, 1.0, 1.0]))
            .mass(1.0)
            .mass_properties(|_| None);
    let sphere = BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).scale([1.0, 1.0, 1_000.0]))
        .mass(1.0)
        .mass_properties(|_| None);
    let cylinder =
        BodyDesc::new(ColliderDesc::new(Shape::cylinder(0.5, 0.5)).scale([1.0, 1_000.0, 1.0]))
            .mass(1.0)
            .mass_properties(|_| None);
    for (name, properties, axis, expected) in [
        ("cube", &cube, 0, (1.0 + 1.0) / 12.0),
        ("sphere", &sphere, 0, (0.25 + 250_000.0) / 5.0),
        ("sphere", &sphere, 5, (0.25 + 0.25) / 5.0),
        ("cylinder", &cylinder, 3, 0.25 / 2.0),
        ("cylinder", &cylinder, 0, (0.75 + 1_000_000.0) / 12.0),
    ] {
        assert!(
            (properties.inertia[axis] / expected - 1.0).abs() < 1e-5,
            "a stretched {name} must keep its closed form on axis {axis}, {expected}, got {}",
            properties.inertia[axis]
        );
        assert_mutual_inverse(properties);
    }
}

#[test]
fn compute_mass_properties_distributes_mass_over_solids() {
    let colliders = [
        ColliderDesc::new(Shape::sphere(1.0)),
        ColliderDesc::new(Shape::sphere(1.0)).offset([0.0, 2.0, 1.0]),
    ];
    let properties = compute_mass_properties(&colliders, MassSource::Fixed(4.0), None, |_| None);
    assert_eq!(properties.com, [0.0, 1.0, 0.5]);
    assert!(properties.inverse_inertia.iter().all(|value| *value >= 0.0));
    assert!(properties.inverse_inertia[0] > 0.0);
}

#[test]
fn scaled_sphere_inertia_uses_axial_transform() {
    let colliders = vec![ColliderDesc::new(Shape::sphere(1.0)).scale([2.0, 1.0, 1.0])];
    let mass = compute_mass_properties(&colliders, MassSource::Fixed(5.0), None, |_| None);
    let m = mass.inverse_inertia;
    let i = [
        1.0 / m[0],
        1.0 / m[1],
        1.0 / m[2],
        1.0 / m[3],
        1.0 / m[4],
        1.0 / m[5],
    ];
    let ellipsoid_y = 5.0 / 5.0 * (4.0 + 1.0);
    assert!(
        (i[3] - ellipsoid_y).abs() < 1e-5,
        "I_y = m/5(b²+c²), got {}",
        i[3]
    );
    let ellipsoid_x = 5.0 / 5.0 * (1.0 + 1.0);
    assert!(
        (i[0] - ellipsoid_x).abs() < 1e-5,
        "I_x = m/5(c²+a²), got {}",
        i[0]
    );
    assert!(
        m[1].abs() < 1e-6 && m[2].abs() < 1e-6,
        "scale keeps axes diagonal"
    );
}

#[test]
fn density_source_scales_mass_and_inertia_by_volume() {
    let colliders = vec![
        ColliderDesc::new(Shape::sphere(0.5)),
        ColliderDesc::new(Shape::sphere(1.0)).offset([0.0, 4.0, 0.0]),
    ];
    let properties = compute_mass_properties(&colliders, MassSource::Density(1.0), None, |_| None);
    let small = 4.0 / 3.0 * std::f32::consts::PI * 0.125;
    let large = 4.0 / 3.0 * std::f32::consts::PI * 1.0;
    let total = small + large;
    let expected_com_y = 4.0 * large / total;
    assert!(
        (properties.com[1] - expected_com_y).abs() < 1e-4,
        "density must center at the volume centroid, got {}",
        properties.com[1]
    );
    assert!((properties.com[0]).abs() < 1e-6);
}
