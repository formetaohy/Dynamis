use dynamis_model::{BodyDesc, ColliderDesc, ConstraintDesc, Shape};
use std::panic::catch_unwind;

#[test]
fn shape_constructors_reject_degenerate_geometry() {
    assert!(catch_unwind(|| Shape::sphere(0.0)).is_err());
    assert!(catch_unwind(|| Shape::sphere(-1.0)).is_err());
    assert!(catch_unwind(|| Shape::cuboid([1.0, 0.0, 1.0])).is_err());
    assert!(catch_unwind(|| Shape::cuboid([1.0, 1.0, -1.0])).is_err());
    assert!(catch_unwind(|| Shape::capsule(0.0, 1.0)).is_err());
    assert!(catch_unwind(|| Shape::capsule(1.0, -0.5)).is_err());
    assert!(catch_unwind(|| Shape::cylinder(0.0, 1.0)).is_err());
    assert!(catch_unwind(|| Shape::cylinder(1.0, -0.5)).is_err());
}

#[test]
fn body_descs_validate_inputs() {
    assert!(catch_unwind(|| BodyDesc::sphere(0.5).orientation([1.0, 1.0, 0.0, 0.0])).is_err());
    assert!(catch_unwind(|| BodyDesc::sphere(0.5).mass(-1.0)).is_err());
    assert!(catch_unwind(|| BodyDesc::sphere(0.5).friction(-0.1)).is_err());
    assert!(
        catch_unwind(|| {
            let mut desc = BodyDesc::sphere(0.5);
            for _ in 0..4 {
                desc = desc.collider(ColliderDesc::new(Shape::sphere(0.1)));
            }
            desc
        })
        .is_err()
    );
    let default = BodyDesc::sphere(0.5);
    assert_eq!(default.collision_group, 1);
    assert_eq!(default.collision_mask, u32::MAX);
    assert_eq!(default.mass, 1.0);
}

#[test]
fn collider_descs_validate_inputs() {
    assert!(
        catch_unwind(|| {
            ColliderDesc::new(Shape::sphere(0.5)).rotation([0.0, 1.0, 1.0, 0.0]);
        })
        .is_err()
    );
    assert!(catch_unwind(|| ColliderDesc::new(Shape::sphere(0.5)).friction(-1.0)).is_err());
}

#[test]
fn constraint_descs_validate_inputs() {
    assert!(catch_unwind(|| ConstraintDesc::distance([0.0; 3], [0.0; 3], -1.0)).is_err());
    assert!(catch_unwind(|| ConstraintDesc::revolute([0.0; 3], [0.0; 3], [0.0; 3])).is_err());
    assert!(catch_unwind(|| ConstraintDesc::prismatic([0.0; 3], [0.0; 3], [0.0; 3])).is_err());
    assert!(
        catch_unwind(
            || ConstraintDesc::revolute([0.0; 3], [0.0; 3], [1.0, 0.0, 0.0]).limit(1.0, 0.0)
        )
        .is_err()
    );
    assert!(
        catch_unwind(|| { ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0).spring(-1.0, 0.5) })
            .is_err()
    );
    assert!(
        catch_unwind(|| { ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0).spring(1.0, -0.5) })
            .is_err()
    );
}

#[test]
fn bounding_radius_matches_shape_extent() {
    assert!((Shape::sphere(0.5).bounding_radius() - 0.5).abs() < 1e-6);
    assert!(
        (Shape::cuboid([1.0, 2.0, 3.0]).bounding_radius() - (1.0f32 + 4.0 + 9.0).sqrt()).abs()
            < 1e-6
    );
    assert!((Shape::capsule(0.3, 1.0).bounding_radius() - (1.0f32 + 0.09).sqrt()).abs() < 1e-6);
    assert!(
        Shape::mesh(dynamis_model::ShapeSourceHandle {
            id: 0,
            generation: 1
        })
        .bounding_radius()
            == 0.0
    );
}
