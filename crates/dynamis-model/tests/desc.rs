use dynamis_model::{
    BodyDesc, BodyHandle, ColliderDesc, ConstraintDesc, FluidMaterial, MaterialCombine,
    PhysicsConfig, Shape, SoftAttachment, SoftBodyDesc, SoftElement, SoftElementKind,
    SoftElementState, SoftMaterial,
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
    let compound = (0..64).fold(BodyDesc::sphere(0.5), |desc, _| {
        desc.collider(ColliderDesc::new(Shape::sphere(0.1)))
    });
    assert_eq!(compound.colliders.len(), 65);
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

#[test]
fn a_fluid_material_bounds_its_spacing_and_support() {
    let material = FluidMaterial::new(0.3, 0.4);
    assert_eq!(material.spacing(), 0.3);
    assert_eq!(material.support(), 0.4);
    assert!(catch_unwind(|| FluidMaterial::new(0.0, 0.4)).is_err());
    assert!(catch_unwind(|| FluidMaterial::new(0.3, 0.3)).is_err());
    let fluid = SoftBodyDesc::fluid(vec![[0.0; 3], [0.3, 0.0, 0.0]], 0.1, material);
    assert!(fluid.elements.is_empty());
    assert_eq!(fluid.radius, 0.1);
    assert_eq!(fluid.fluid.unwrap().support(), 0.4);
    let crowded =
        catch_unwind(|| SoftBodyDesc::fluid(vec![[0.0; 3]], 0.2, FluidMaterial::new(0.3, 0.4)));
    assert!(
        crowded.is_err(),
        "a fluid particle must fit its rest spacing"
    );
}

#[test]
fn a_soft_element_bounds_its_strength() {
    let yielding = SoftElement::distance(0, 1, 1.0)
        .yielding(0.2, 0.5)
        .fracturing(0.4);
    assert_eq!(yielding.yield_strain_of(), 0.2);
    assert_eq!(yielding.break_strain_of(), 0.4);
    assert_eq!(yielding.plastic_flow_of(), 0.5);
    assert!(yielding.carries_strength());
    assert!(!SoftElement::distance(0, 1, 1.0).carries_strength());
    let brittle = SoftElement::area(0, 1, 2, 0.5).fracturing(0.05);
    assert!(brittle.carries_strength());
    assert!(brittle.yield_strain_of().is_infinite());
    assert!(catch_unwind(|| SoftElement::distance(0, 1, 1.0).yielding(-0.1, 0.5)).is_err());
    assert!(catch_unwind(|| SoftElement::distance(0, 1, 1.0).yielding(0.1, 1.5)).is_err());
    assert!(catch_unwind(|| SoftElement::distance(0, 1, 1.0).fracturing(-0.1)).is_err());
}

#[test]
fn a_soft_material_spreads_its_strength_over_its_elements() {
    let material = SoftMaterial::new(0.01, 0.02, 0.03, 0.04)
        .yielding(0.2, 0.25)
        .fracturing(0.4);
    let desc = SoftBodyDesc::cloth([3, 3], 1.0, material);
    assert!(desc.carries_strength());
    assert!(desc.elements.iter().all(|element| {
        element.yield_strain_of() == 0.2
            && element.break_strain_of() == 0.4
            && element.plastic_flow_of() == 0.25
    }));
    assert!(
        desc.elements
            .iter()
            .any(|element| element.kind() == SoftElementKind::Area)
    );
    let elastic = SoftBodyDesc::cloth([3, 3], 1.0, SoftMaterial::rigid());
    assert!(!elastic.carries_strength());
    let overridden = desc.clone().yielding(0.05, 1.0).fracturing(0.1);
    assert!(
        overridden.elements.iter().all(|element| {
            element.yield_strain_of() == 0.05 && element.break_strain_of() == 0.1
        })
    );
    assert_eq!(
        SoftElementState::new(
            SoftElementKind::Distance,
            [3, 4, u32::MAX, u32::MAX],
            0.5,
            true
        )
        .participants()
        .collect::<Vec<_>>(),
        vec![3, 4]
    );
}

#[test]
fn a_soft_attachment_binds_one_particle_to_a_body() {
    let body = BodyHandle {
        id: 7,
        generation: 2,
    };
    let anchor = SoftAttachment::new(1, body, [0.5, 0.0, 0.0]);
    assert_eq!(anchor.particle(), 1);
    assert_eq!(anchor.body(), body);
    assert_eq!(anchor.local(), [0.5, 0.0, 0.0]);
    let net = SoftBodyDesc::net(vec![[0.0; 3], [0.0, 1.0, 0.0]], vec![[0, 1]]);
    let attached = net.clone().attach(anchor);
    assert_eq!(attached.attachments, vec![anchor]);
    assert!(attached.attachments[0].body() == body);
    assert!(
        catch_unwind(AssertUnwindSafe(|| attached.clone().attach(anchor))).is_err(),
        "a soft particle carries at most one attachment"
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            net.clone().attach(SoftAttachment::new(2, body, [0.0; 3]))
        }))
        .is_err(),
        "a soft attachment must reference a live particle"
    );
}
