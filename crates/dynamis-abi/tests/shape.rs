use dynamis_abi::{
    SHAPE_CAPSULE, SHAPE_CUBOID, SHAPE_CYLINDER, SHAPE_HEIGHTFIELD, SHAPE_HULL, SHAPE_MESH,
    SHAPE_NONE, SHAPE_PLANE, SHAPE_ROLE_COUNT, SHAPE_ROLES, SHAPE_SPHERE, SHAPE_TRIANGLE,
    ShapePath, ShapeRole, ShapeScale, constants_wgsl, shape_code, shape_source_handle,
};
use dynamis_model::{Shape, ShapeSourceHandle};

fn source() -> ShapeSourceHandle {
    ShapeSourceHandle {
        id: 4,
        generation: 2,
    }
}

fn every_shape() -> Vec<(Shape, u32)> {
    vec![
        (Shape::sphere(0.5), SHAPE_SPHERE),
        (Shape::cuboid([0.5, 0.25, 0.125]), SHAPE_CUBOID),
        (Shape::capsule(0.25, 0.5), SHAPE_CAPSULE),
        (Shape::cylinder(0.25, 0.5), SHAPE_CYLINDER),
        (Shape::hull(source()), SHAPE_HULL),
        (Shape::mesh(source()), SHAPE_MESH),
        (Shape::height_field(source()), SHAPE_HEIGHTFIELD),
        (Shape::plane(), SHAPE_PLANE),
    ]
}

#[test]
fn every_device_shape_code_carries_exactly_one_declared_role() {
    assert_eq!(
        SHAPE_ROLE_COUNT,
        SHAPE_PLANE + 1,
        "the role table must cover every shape code the device declares"
    );
    assert_eq!(SHAPE_ROLES.len(), SHAPE_ROLE_COUNT as usize);
    for (index, role) in SHAPE_ROLES.iter().enumerate() {
        assert_eq!(
            role.code, index as u32,
            "the role table must name every shape code once, in order"
        );
        assert_eq!(
            ShapeRole::of(role.code),
            *role,
            "a code must answer the role the table declares for it"
        );
        if role.code == SHAPE_NONE {
            assert_eq!(
                role.path,
                ShapePath::Vacant,
                "the vacant code must belong to no collision path"
            );
            continue;
        }
        let convex = role.path == ShapePath::Convex;
        assert_ne!(
            convex,
            role.path == ShapePath::WorldGeometry,
            "a real shape code must belong to exactly one collision path"
        );
        assert!(!convex || role.scale != ShapeScale::NotScalable);
        assert!(role.path != ShapePath::WorldGeometry || role.scale != ShapeScale::UniformFolded);
        assert!(!role.analytic || convex);
        assert!(!role.source || role.scale == ShapeScale::InRecord);
    }
}

#[test]
fn every_shape_description_answers_a_declared_role_and_source() {
    let mut codes = Vec::new();
    for (shape, code) in every_shape() {
        let role = ShapeRole::of_shape(&shape);
        assert_eq!(role.code, code, "a shape must answer its declared code");
        assert_eq!(shape_code(&shape), code);
        assert_eq!(role, ShapeRole::of(code));
        assert!(role.convex() || role.world_geometry());
        assert!(!role.triangle_scene() || role.world_geometry());
        assert_eq!(
            shape_source_handle(&shape).is_some(),
            role.source,
            "the source handle and the declared source must agree"
        );
        codes.push(code);
    }
    codes.sort_unstable();
    codes.dedup();
    assert_eq!(
        codes.len(),
        every_shape().len(),
        "every shape description must answer its own code"
    );
    assert!(
        !codes.contains(&SHAPE_NONE) && !codes.contains(&SHAPE_TRIANGLE),
        "a description never answers the vacant code or the scene triangle"
    );
}

type Predicate = (&'static str, fn(ShapeRole) -> bool);

#[test]
fn a_device_predicate_names_exactly_the_codes_its_role_declares() {
    let source = constants_wgsl();
    let predicates: [Predicate; 4] = [
        ("shape_world_geometry", ShapeRole::world_geometry),
        ("shape_triangle_scene", ShapeRole::triangle_scene),
        ("shape_source", |role| role.source),
        ("shape_analytic", |role| role.analytic),
    ];
    for (name, holds) in predicates {
        assert_eq!(
            declared_codes(&source, name),
            selected_codes(holds),
            "the emitted {name} must name the codes its role declares"
        );
    }
    let world_geometry = declared_codes(&source, "shape_world_geometry");
    let source_codes = declared_codes(&source, "shape_source");
    assert!(
        !world_geometry.contains(&SHAPE_TRIANGLE),
        "a scene triangle answers the convex path, not the world geometry path"
    );
    assert!(
        source_codes.contains(&SHAPE_TRIANGLE),
        "a scene triangle still owns a source"
    );
}

fn selected_codes(holds: fn(ShapeRole) -> bool) -> Vec<u32> {
    SHAPE_ROLES
        .iter()
        .filter(|role| holds(**role))
        .map(|role| role.code)
        .collect()
}

fn declared_codes(source: &str, name: &str) -> Vec<u32> {
    let head = format!("fn {name}(kind: u32) -> bool {{ return ");
    let start = source
        .find(&head)
        .unwrap_or_else(|| panic!("the shader constants must declare {name}"));
    let body = &source[start + head.len()..];
    let end = body.find(';').expect("a predicate ends with its statement");
    let mut codes = Vec::new();
    for term in body[..end].split("||") {
        let term = term.trim();
        if term == "false" {
            continue;
        }
        let code = term
            .strip_prefix("kind == ")
            .and_then(|code| code.strip_suffix('u'))
            .unwrap_or_else(|| panic!("a predicate term reads as a code: {term}"))
            .parse::<u32>()
            .expect("a predicate term names a code");
        codes.push(code);
    }
    codes
}

#[test]
fn a_collider_scale_reaches_the_record_its_role_declares() {
    let uniform = [2.0; 3];
    let mixed = [2.0, 1.0, 0.5];
    assert_eq!(
        ShapeScale::InDimensions.dimension_factor(mixed),
        mixed,
        "a cuboid folds its whole scale into its half extents"
    );
    assert_eq!(ShapeScale::InDimensions.record_scale(mixed), [1.0; 3]);
    assert_eq!(ShapeScale::UniformFolded.dimension_factor(uniform), uniform);
    assert_eq!(ShapeScale::UniformFolded.record_scale(uniform), [1.0; 3]);
    assert_eq!(
        ShapeScale::UniformFolded.dimension_factor(mixed),
        [1.0; 3],
        "a non-uniform scale must stay in the record"
    );
    assert_eq!(ShapeScale::UniformFolded.record_scale(mixed), mixed);
    assert_eq!(ShapeScale::InRecord.dimension_factor(mixed), [1.0; 3]);
    assert_eq!(ShapeScale::InRecord.record_scale(mixed), mixed);
    assert_eq!(ShapeScale::NotScalable.dimension_factor(mixed), [1.0; 3]);
    assert_eq!(ShapeScale::NotScalable.record_scale(mixed), [1.0; 3]);

    for (shape, _) in every_shape() {
        let role = ShapeRole::of_shape(&shape);
        let collider = dynamis_model::ColliderDesc::new(shape).scale(mixed);
        let record = dynamis_abi::ColliderRecord::build(&collider, 0, 0);
        assert_eq!(record.kind, role.code);
        assert_eq!(record.scale, role.scale.record_scale(mixed));
    }
}
