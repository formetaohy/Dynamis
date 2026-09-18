use dynamis_model::{
    BodyDesc, BodyHandle, ColliderDesc, CollisionFilter, ConstraintDesc, ConstraintKind, DofDesc,
    FluidMaterial, MaterialCombine, PhysicsConfig, Shape, SoftAttachment, SoftBodyDesc,
    SoftElement, SoftElementKind, SoftElementState, SoftMaterial,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn a_config_declares_the_domain_of_every_field_it_carries() {
    let config = PhysicsConfig::default();
    config.assert_valid();
    type Break = fn(&mut PhysicsConfig);
    let refused: [(&str, Break); 13] = [
        ("gravity", |config| config.gravity = [f32::NAN, 0.0, 0.0]),
        ("damping", |config| config.damping = f32::INFINITY),
        ("angular damping", |config| config.angular_damping = -1.0),
        ("substeps", |config| config.substeps = 0),
        ("solve iterations", |config| config.solve_iterations = 0),
        ("position iterations", |config| {
            config.position_iterations = 0
        }),
        ("soft substeps", |config| config.soft_substeps = 0),
        ("soft iterations", |config| config.soft_iterations = 0),
        ("relaxation", |config| config.relaxation = 0.0),
        ("slop", |config| config.slop = f32::NAN),
        ("contact margin", |config| config.contact_margin = -1.0),
        ("velocity limit", |config| config.max_velocity = 0.0),
        ("sleep time", |config| config.sleep_time = f32::INFINITY),
    ];
    for (field, break_it) in refused {
        let mut broken = config;
        break_it(&mut broken);
        assert!(
            catch_unwind(|| broken.assert_valid()).is_err(),
            "{field} must be refused outside its domain"
        );
    }
    let resting = PhysicsConfig {
        damping: 0.0,
        angular_damping: 0.0,
        slop: 0.0,
        contact_margin: 0.0,
        restitution_threshold: 0.0,
        sleep_velocity: 0.0,
        sleep_angular_velocity: 0.0,
        settle_velocity: 0.0,
        relaxation: 1.0,
        ..PhysicsConfig::default()
    };
    resting.assert_valid();
}

#[test]
fn a_description_declares_the_domain_of_every_field_it_carries() {
    let mut body = BodyDesc::sphere(0.5);
    body.assert_valid();
    body.position = [0.0, f32::NAN, 0.0];
    assert!(catch_unwind(|| body.assert_valid()).is_err());
    body.position = [0.0; 3];
    body.velocity = [f32::INFINITY; 3];
    assert!(catch_unwind(|| body.assert_valid()).is_err());
    body.velocity = [0.0; 3];
    body.com = Some([f32::NAN; 3]);
    assert!(catch_unwind(|| body.assert_valid()).is_err());
    body.com = None;
    body.colliders.clear();
    assert!(catch_unwind(|| body.assert_valid()).is_err());
    let mut collider = ColliderDesc::new(Shape::sphere(0.5));
    collider.assert_valid();
    collider.restitution = f32::NAN;
    assert!(catch_unwind(|| collider.assert_valid()).is_err());
    collider.restitution = 1.0;
    collider.contact_frequency = 0.0;
    assert!(catch_unwind(|| collider.assert_valid()).is_err());
    collider.contact_frequency = f32::INFINITY;
    collider.assert_valid();
    let dimensionless = Shape::Sphere {
        radius: f32::INFINITY,
    };
    assert!(catch_unwind(|| dimensionless.assert_valid()).is_err());
    let degenerate = Shape::Cuboid {
        half_extents: [1.0, 0.0, 1.0],
    };
    assert!(catch_unwind(|| degenerate.assert_valid()).is_err());
    assert!(
        catch_unwind(|| ConstraintDesc::revolute([0.0; 3], [0.0; 3], [f32::NAN, 0.0, 0.0]))
            .is_err()
    );
    assert!(catch_unwind(|| ConstraintDesc::distance([f32::NAN; 3], [0.0; 3], 1.0)).is_err());
    let mut soft = SoftBodyDesc::new(vec![[0.0; 3]], Vec::new());
    soft.assert_valid();
    soft.inverse_masses = vec![f32::NAN];
    assert!(catch_unwind(|| soft.assert_valid()).is_err());
    soft.inverse_masses = Vec::new();
    assert!(catch_unwind(|| soft.assert_valid()).is_err());
    soft.inverse_masses = vec![1.0];
    soft.velocity = [0.0, f32::INFINITY, 0.0];
    assert!(catch_unwind(|| soft.assert_valid()).is_err());
    let mut surface = dynamis_model::SurfaceDesc::new();
    surface.assert_valid();
    surface.friction = f32::NAN;
    assert!(catch_unwind(|| surface.assert_valid()).is_err());
}

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
    assert_eq!(default.filter.group(), 1);
    assert_eq!(default.filter.mask(), u32::MAX);
    assert_eq!(default.filter, CollisionFilter::DEFAULT);
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
fn a_joint_declares_only_the_properties_its_kind_carries() {
    let anchor = || ([0.0; 3], [0.0; 3]);
    let axis = || [0.0, 0.0, 1.0];
    type Refused = (&'static str, fn() -> ConstraintDesc);
    let unsupported: [Refused; 17] = [
        ("distance limit", || {
            ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0).limit(0.0, 1.0)
        }),
        ("distance motor", || {
            ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0).motor(1.0)
        }),
        ("distance swing", || {
            ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0).swing(0.1, 0.1)
        }),
        ("ball spring", || {
            ConstraintDesc::ball([0.0; 3], [0.0; 3]).spring(1.0, 0.5)
        }),
        ("ball motor", || {
            ConstraintDesc::ball([0.0; 3], [0.0; 3]).motor(1.0)
        }),
        ("revolute spring", || {
            ConstraintDesc::revolute([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]).spring(1.0, 0.5)
        }),
        ("prismatic swing", || {
            ConstraintDesc::prismatic([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]).swing(0.1, 0.1)
        }),
        ("cone limit", || {
            ConstraintDesc::cone([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.5)
                .limit(0.0, 1.0)
        }),
        ("gear motor", || {
            ConstraintDesc::gear([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 2.0).motor(1.0)
        }),
        ("pulley spring", || {
            ConstraintDesc::pulley([0.0; 3], [0.0; 3], [0.0; 3], [1.0, 0.0, 0.0], 1.0)
                .spring(1.0, 0.5)
        }),
        ("fixed limit", || {
            ConstraintDesc::fixed([0.0; 3], [0.0; 3]).limit(-1.0, 1.0)
        }),
        ("six dof limit", || {
            ConstraintDesc::six_dof([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]).limit(-1.0, 1.0)
        }),
        ("six dof swing", || {
            ConstraintDesc::six_dof([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]).swing(0.1, 0.1)
        }),
        ("six dof spring", || {
            ConstraintDesc::six_dof([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]).spring(1.0, 0.5)
        }),
        ("ball dofs", || {
            ConstraintDesc::ball([0.0; 3], [0.0; 3]).dofs([DofDesc::locked(); 6])
        }),
        ("fixed dof", || {
            ConstraintDesc::fixed([0.0; 3], [0.0; 3]).dof(0, DofDesc::locked())
        }),
        ("gear axis", || {
            ConstraintDesc::gear([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 2.0).axis([1.0, 0.0, 0.0])
        }),
    ];
    for (property, build) in unsupported {
        assert!(
            catch_unwind(build).is_err(),
            "{property} must be rejected at the joint it does not belong to"
        );
    }
    let carried = [
        ConstraintDesc::ball(anchor().0, anchor().1)
            .axis(axis())
            .limit(-0.5, 0.5)
            .swing(0.7, 0.9),
        ConstraintDesc::distance(anchor().0, anchor().1, 1.0).spring(1.0, 0.5),
        ConstraintDesc::revolute(anchor().0, anchor().1, axis())
            .limit(-0.5, 0.5)
            .motor(1.0),
        ConstraintDesc::prismatic(anchor().0, anchor().1, axis())
            .limit(-0.5, 0.5)
            .motor(1.0),
        ConstraintDesc::fixed(anchor().0, anchor().1),
        ConstraintDesc::gear([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 2.0),
        ConstraintDesc::pulley(anchor().0, anchor().1, [0.0; 3], [1.0, 0.0, 0.0], 1.0),
        ConstraintDesc::cone(anchor().0, anchor().1, axis(), axis(), 0.5),
        ConstraintDesc::six_dof(anchor().0, anchor().1, axis()).dofs([DofDesc::locked(); 6]),
    ];
    for desc in carried {
        assert_eq!(desc.kind(), desc.data().kind());
        assert_eq!(
            desc.dofs_of().is_some(),
            desc.kind() == ConstraintKind::SixDof
        );
    }
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
    assert_eq!(motor.motor_of().unwrap().target_velocity(), 3.0);
    assert_eq!(motor.motor_of().unwrap().max_force(), 50.0);
    let broken = ConstraintDesc::ball([0.0; 3], [0.0; 3]).break_threshold(100.0, 10.0);
    assert_eq!(broken.break_threshold_of().unwrap().force, 100.0);
    let swiveling = ConstraintDesc::ball([0.0; 3], [0.0; 3])
        .limit(-0.5, 0.5)
        .swing(0.7, 0.9);
    assert_eq!(swiveling.swing_of().unwrap().swing_a, 0.7);
    assert_eq!(swiveling.limit_of().unwrap().max, 0.5);
    let gear = ConstraintDesc::gear([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 2.5);
    match gear.data() {
        dynamis_model::ConstraintData::Gear {
            axis_a,
            axis_b,
            ratio,
        } => {
            assert_eq!(*axis_a, [0.0, 1.0, 0.0]);
            assert_eq!(*axis_b, [0.0, 0.0, 1.0]);
            assert_eq!(*ratio, 2.5);
        }
        other => panic!("a gear joint must carry a gear payload, got {other:?}"),
    }
    let pulley = ConstraintDesc::pulley(
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
        3.0,
    );
    match pulley.data() {
        dynamis_model::ConstraintData::Pulley {
            length, fixed_a, ..
        } => {
            assert_eq!(*length, 3.0);
            assert_eq!(*fixed_a, [0.0, 2.0, 0.0]);
        }
        other => panic!("a pulley joint must carry a pulley payload, got {other:?}"),
    }
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
