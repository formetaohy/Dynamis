use dynamis_model::{
    BodyDesc, ColliderDesc, ConstraintDesc, MaterialCombine, PhysicsConfig, Shape,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

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
            for _ in 0..16 {
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
fn plane_shape_and_scaled_colliders_validate() {
    let plane = Shape::plane();
    assert!(plane.is_world_geometry());
    assert!(!plane.is_convex());
    let scaled = ColliderDesc::new(Shape::sphere(0.5)).scale([2.0, 1.0, 1.0]);
    assert_eq!(scaled.scale, [2.0, 1.0, 1.0]);
    let invalid = catch_unwind(AssertUnwindSafe(|| {
        ColliderDesc::new(Shape::sphere(0.5)).scale([0.0, 1.0, 1.0])
    }));
    assert!(invalid.is_err(), "scale must be strictly positive");
}

#[test]
fn combine_modes_and_constraint_apis() {
    let config = PhysicsConfig {
        friction_combine: MaterialCombine::Min,
        restitution_combine: MaterialCombine::Average,
        ..PhysicsConfig::default()
    };
    assert_eq!(config.friction_combine, MaterialCombine::Min);
    assert_eq!(
        PhysicsConfig::default().restitution_combine,
        MaterialCombine::Max
    );
    let motor = ConstraintDesc::revolute([0.0; 3], [0.0; 3], [0.0, 1.0, 0.0])
        .motor(3.0)
        .motor_force(50.0);
    assert_eq!(motor.motor.unwrap().target_velocity, 3.0);
    assert_eq!(motor.motor.unwrap().max_force, 50.0);
    let broken = ConstraintDesc::ball([0.0; 3], [0.0; 3]).break_threshold(100.0, 10.0);
    assert_eq!(broken.break_threshold.unwrap().force, 100.0);
    let swiveling = ConstraintDesc::ball([0.0; 3], [0.0; 3])
        .limit(-0.5, 0.5)
        .swing(0.7, 0.9);
    assert_eq!(swiveling.swing.unwrap().swing_a, 0.7);
    let gear = ConstraintDesc::gear([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 2.5);
    assert_eq!(gear.gear_ratio, 2.5);
    assert_eq!(gear.axis_b, [0.0, 0.0, 1.0]);
    let pulley = ConstraintDesc::pulley(
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
        3.0,
    );
    assert_eq!(pulley.rest_length, 3.0);
    assert_eq!(pulley.pulley_fixed_a, [0.0, 2.0, 0.0]);
    let invalid = catch_unwind(AssertUnwindSafe(|| {
        ConstraintDesc::gear([0.0; 3], [0.0, 1.0, 0.0], 1.0)
    }));
    assert!(invalid.is_err(), "gear axis must be non-zero");
    let invalid = catch_unwind(AssertUnwindSafe(|| {
        ConstraintDesc::pulley([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], 0.0)
    }));
    assert!(invalid.is_err(), "pulley length must be positive");
}
