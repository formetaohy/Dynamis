use super::common::{DT, gravity_config, observed_world, static_config};
use dynamis_model::{
    BodyDesc, CharacterDesc, ColliderDesc, CollisionFilter, PhysicsConfig, Shape, VehicleDesc,
    WheelDesc,
};
use dynamis_world::World;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn mesh() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    (
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
    )
}

fn refuses(what: &str, world: &mut World, attempt: impl FnOnce(&mut World)) {
    assert!(
        catch_unwind(AssertUnwindSafe(|| attempt(world))).is_err(),
        "{what} must be refused"
    );
}

fn settle(world: &mut World, frames: usize) {
    for _ in 0..frames {
        world.step(DT);
    }
    world.wait();
}

type Attempt = Box<dyn FnOnce(&mut World)>;

#[test]
fn a_simulating_body_refuses_world_geometry() {
    let mut world = observed_world(gravity_config());
    let (vertices, triangles) = mesh();
    let source = world.add_mesh(&vertices, &triangles, None);
    let world_geometry =
        world.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(source))).mass(0.0));
    let mixed = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::mesh(source)))
            .collider(ColliderDesc::new(Shape::sphere(0.5)))
            .mass(0.0),
    );
    let solid = world.spawn(BodyDesc::sphere(0.5).position([0.0, 3.0, 0.0]));
    settle(&mut world, 1);
    let held_geometry = world.read_state(world_geometry);
    let held_mixed = world.read_state(mixed);

    let raws: Vec<(&'static str, Attempt)> = vec![
        (
            "a static mesh given a mass",
            Box::new(move |world: &mut World| world.set_mass(world_geometry, 1.0)),
        ),
        (
            "a mesh carried by a body a density would weigh",
            Box::new(move |world: &mut World| world.set_density(mixed, 1.0)),
        ),
        (
            "a mesh declared as a simulating body",
            Box::new(|world: &mut World| {
                let (vertices, triangles) = mesh();
                let source = world.add_mesh(&vertices, &triangles, None);
                world.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(source))).mass(1.0));
            }),
        ),
        (
            "a plane declared as a simulating body",
            Box::new(|world: &mut World| {
                world.spawn(BodyDesc::new(ColliderDesc::new(Shape::plane())).mass(1.0));
            }),
        ),
        (
            "a mesh shape on a simulating body",
            Box::new(move |world: &mut World| world.set_shape(solid, Shape::mesh(source))),
        ),
        (
            "a plane shape on a simulating body",
            Box::new(move |world: &mut World| world.set_shape(solid, Shape::plane())),
        ),
        (
            "a mesh collider added to a simulating body",
            Box::new(move |world: &mut World| {
                world.add_collider(solid, ColliderDesc::new(Shape::mesh(source)))
            }),
        ),
        (
            "a replaced collider of a simulating body carrying a mesh",
            Box::new(move |world: &mut World| {
                world.set_collider(solid, 0, ColliderDesc::new(Shape::mesh(source)))
            }),
        ),
    ];

    for (what, attempt) in raws {
        refuses(what, &mut world, attempt);
        let probe = catch_unwind(AssertUnwindSafe(|| world.set_mass(solid, 1.0)));
        assert!(
            probe.is_ok(),
            "{what} left world geometry on the simulating body it was refused"
        );
        settle(&mut world, 2);
        let geometry = world.read_state(world_geometry);
        assert_eq!(
            geometry.position, held_geometry.position,
            "{what} moved the world geometry it was refused"
        );
        assert_eq!(
            geometry.inverse_mass, held_geometry.inverse_mass,
            "{what} reweighed the world geometry it was refused"
        );
        let compound = world.read_state(mixed);
        assert_eq!(
            compound.position, held_mixed.position,
            "{what} moved the compound it was refused"
        );
        assert_eq!(
            compound.inverse_mass, held_mixed.inverse_mass,
            "{what} reweighed the compound it was refused"
        );
        assert!(
            world.refusals().is_empty(),
            "{what} refused work the device cannot lose"
        );
    }
}

#[test]
fn an_actor_body_keeps_the_kind_its_actor_declares() {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([10.0, 0.5, 10.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let character = world.add_character([0.0, 1.0, 0.0], CharacterDesc::default());
    let character_body = world.character_body(character);
    let vehicle = world.add_vehicle(VehicleDesc::new(
        BodyDesc::cuboid([0.9, 0.3, 1.8]).position([4.0, 0.8, 0.0]),
        vec![
            WheelDesc::new([0.8, -0.3, 0.9], 0.3).driving(),
            WheelDesc::new([-0.8, -0.3, 0.9], 0.3),
        ],
    ));
    let chassis = world.vehicle_body(vehicle);
    settle(&mut world, 60);
    let held_character = world.read_state(character_body);
    let held_chassis = world.read_state(chassis);

    let raws: Vec<(&'static str, Attempt)> = vec![
        (
            "a character body declared dynamic",
            Box::new(move |world: &mut World| world.set_kinematic(character_body, false)),
        ),
        (
            "a character collider reshaped",
            Box::new(move |world: &mut World| world.set_shape(character_body, Shape::sphere(0.5))),
        ),
        (
            "a character collider replaced",
            Box::new(move |world: &mut World| {
                world.set_collider(character_body, 0, ColliderDesc::new(Shape::sphere(0.5)))
            }),
        ),
        (
            "a character collider added",
            Box::new(move |world: &mut World| {
                world.add_collider(character_body, ColliderDesc::new(Shape::sphere(0.2)))
            }),
        ),
        (
            "a vehicle chassis frozen",
            Box::new(move |world: &mut World| world.set_mass(chassis, 0.0)),
        ),
        (
            "a vehicle chassis driven by the host",
            Box::new(move |world: &mut World| world.set_kinematic(chassis, true)),
        ),
    ];

    for (what, attempt) in raws {
        refuses(what, &mut world, attempt);
        let probe = catch_unwind(AssertUnwindSafe(|| {
            world.set_kinematic(character_body, true);
            world.set_mass(chassis, 1.0);
        }));
        assert!(
            probe.is_ok(),
            "{what} left the declaration of the actor that owns the body broken"
        );
        settle(&mut world, 2);
        assert_eq!(
            world.read_state(character_body).inverse_mass,
            held_character.inverse_mass,
            "{what} reweighed the character body"
        );
        assert_eq!(
            world.read_state(chassis).inverse_mass,
            held_chassis.inverse_mass,
            "{what} reweighed the vehicle chassis"
        );
        assert!(
            world.inspect_character_state(character).grounded,
            "{what} stopped the character from standing"
        );
        assert!(
            (world.read_state(character_body).position[1] - held_character.position[1]).abs() < 0.2,
            "{what} dropped the character body"
        );
        assert!(
            world.refusals().is_empty(),
            "{what} refused work the device cannot lose"
        );
    }
}

#[test]
fn a_world_geometry_body_takes_every_kind_the_host_may_declare() {
    let mut world = observed_world(gravity_config());
    let (vertices, triangles) = mesh();
    let source = world.add_mesh(&vertices, &triangles, None);
    let floor = world.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(source))).mass(0.0));

    world.set_mass(floor, 0.0);
    world.set_density(floor, 3.0);
    settle(&mut world, 4);
    assert_eq!(
        world.read_state(floor).position,
        [0.0; 3],
        "a static mesh must stay where the host declared it"
    );

    world.set_kinematic(floor, true);
    world.set_velocity(floor, [1.0, 0.0, 0.0]);
    world.set_mass(floor, 4.0);
    settle(&mut world, 10);
    let moved = world.read_state(floor).position[0];
    assert!(
        (moved - DT * 10.0).abs() < 1e-4,
        "a kinematic mesh must advance by the motion its host declares, got {moved}"
    );

    world.set_mass(floor, 0.0);
    world.set_kinematic(floor, false);
    let held = world.read_state(floor).position[0];
    settle(&mut world, 10);
    assert_eq!(
        world.read_state(floor).position[0],
        held,
        "a mesh the host stops driving must stay where it stopped"
    );
}

#[test]
fn an_actor_body_takes_every_edit_its_actor_does_not_own() {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([10.0, 0.5, 10.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let character = world.add_character([0.0, 1.0, 0.0], CharacterDesc::default());
    let character_body = world.character_body(character);
    let vehicle = world.add_vehicle(VehicleDesc::new(
        BodyDesc::cuboid([0.9, 0.3, 1.8]).position([4.0, 0.8, 0.0]),
        vec![
            WheelDesc::new([0.8, -0.3, 0.9], 0.3).driving(),
            WheelDesc::new([-0.8, -0.3, 0.9], 0.3),
        ],
    ));
    let chassis = world.vehicle_body(vehicle);

    world.set_collision_filter(character_body, CollisionFilter::DEFAULT);
    world.set_mass(character_body, 3.0);
    world.set_ccd(chassis, true);
    world.set_mass(chassis, 2.0);
    settle(&mut world, 10);
    assert!(
        world.refusals().is_empty(),
        "an edit the actor does not own must reach the simulation"
    );
}

#[test]
fn a_simulating_body_may_take_every_solid_collider() {
    let mut world = observed_world(PhysicsConfig {
        gravity: [0.0; 3],
        ..PhysicsConfig::default()
    });
    let (vertices, triangles) = mesh();
    let hull = world.add_hull(&vertices, &triangles);
    let body = world.spawn(BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5))).density(2.0));
    let single = world.read_state(body).inverse_mass;
    world.add_collider(
        body,
        ColliderDesc::new(Shape::hull(hull)).offset([2.0, 0.0, 0.0]),
    );
    assert!(
        world.read_state(body).inverse_mass < single,
        "a solid collider must weigh the body it joins"
    );
    world.set_mass(body, 2.0);
    world.set_shape(body, Shape::sphere(0.4));
    settle(&mut world, 2);
    assert!(
        world.refusals().is_empty(),
        "a solid collider must reach the simulation"
    );
}

fn cube(extent: f32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let h = extent * 0.5;
    let vertices = vec![
        [-h, -h, -h],
        [h, -h, -h],
        [h, h, -h],
        [-h, h, -h],
        [-h, -h, h],
        [h, -h, h],
        [h, h, h],
        [-h, h, h],
    ];
    let triangles = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
        [1, 2, 6],
        [1, 6, 5],
        [0, 4, 7],
        [0, 7, 3],
    ];
    (vertices, triangles)
}

#[test]
fn a_source_update_reweighs_every_body_that_reads_it() {
    let mut world = observed_world(static_config());
    let (vertices, triangles) = cube(2.0);
    let source = world.add_hull(&vertices, &triangles);
    let body = world.spawn(BodyDesc::new(ColliderDesc::new(Shape::hull(source))).density(1.0));
    settle(&mut world, 1);
    let before = world.read_state(body).inverse_mass;
    let (wide, wide_triangles) = cube(4.0);
    world.update_mesh(source, &wide, &wide_triangles, None);
    let after = world.read_state(body).inverse_mass;
    assert!(
        (after * 8.0 - before).abs() < 1e-6,
        "a hull that doubles its extent must weigh eight times as much: {before} -> {after}"
    );
    world.apply_impulse(body, [1.0, 0.0, 0.0]);
    settle(&mut world, 1);
    let velocity = world.read_state(body).velocity[0];
    assert!(
        (velocity - 1.0 / 64.0).abs() < 1e-6,
        "the simulation must answer the updated hull mass, got velocity {velocity}"
    );
}

#[test]
fn a_source_is_released_by_every_reader_it_held() {
    let mut world = observed_world(static_config());
    let (vertices, triangles) = cube(2.0);
    let source = world.add_hull(&vertices, &triangles);
    let first = world.spawn(BodyDesc::new(ColliderDesc::new(Shape::hull(source))).mass(0.0));
    let second = world.spawn(BodyDesc::new(ColliderDesc::new(Shape::hull(source))).mass(0.0));
    world.add_collider(
        second,
        ColliderDesc::new(Shape::hull(source)).offset([5.0, 0.0, 0.0]),
    );
    world.remove_collider(second, 1);
    world.remove(first);
    world.remove(second);
    world.remove_shape(source);
    settle(&mut world, 1);
    assert!(
        world.refusals().is_empty(),
        "a source no body reads must be removable"
    );
}
