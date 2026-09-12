use super::common::{DT, flat_mesh_floor, gravity_config, new_world, settle};
use dynamis_model::{BodyDesc, BodyHandle, ColliderDesc, Shape};
use dynamis_world::{ContactManifold, World};

fn cube_mesh() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let vertices = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let faces = vec![
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
    (vertices, faces)
}

fn pair_matches(manifold: &ContactManifold, first: BodyHandle, second: BodyHandle) -> bool {
    (manifold.first == first && manifold.second == second)
        || (manifold.first == second && manifold.second == first)
}

fn features_of(world: &mut World, first: BodyHandle, second: BodyHandle) -> Vec<u32> {
    let mut features = world
        .contact_manifolds()
        .into_iter()
        .filter(|manifold| pair_matches(manifold, first, second))
        .flat_map(|manifold| manifold.points.into_iter().map(|point| point.feature))
        .collect::<Vec<_>>();
    features.sort_unstable();
    features
}

fn wide_floor(world: &mut World, friction: f32) -> BodyHandle {
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::cuboid([16.0, 0.5, 16.0])).friction(friction))
            .position([0.0, -0.5, 0.0])
            .mass(0.0),
    )
}

#[test]
fn sliding_contacts_hold_their_features() {
    let mut world = new_world(gravity_config());
    let floor = wide_floor(&mut world, 0.0);
    let cube = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::cuboid([0.5, 0.5, 0.5])).friction(0.0))
            .position([0.0, 0.5, 0.0]),
    );
    settle(&mut world, 30);
    let resting = features_of(&mut world, cube, floor);
    assert!(
        !resting.is_empty(),
        "a resting box must hold contact points"
    );

    world.set_velocity(cube, [3.0, 0.0, 0.0]);
    let mut travel = 0.0;
    let mut previous = world.read_state(cube).position[0];
    for frame in 0..40usize {
        world.step(DT);
        world.wait();
        let position = world.read_state(cube).position[0];
        travel += position - previous;
        previous = position;
        assert_eq!(
            features_of(&mut world, cube, floor),
            resting,
            "contact features must survive sliding at frame {frame}"
        );
    }
    assert!(
        travel > 1.5,
        "the box must slide far enough to exercise clipping, travelled {travel}"
    );
}

#[test]
fn mesh_contacts_hold_their_triangle_features() {
    let mut world = new_world(gravity_config());
    let floor = flat_mesh_floor(&mut world);
    let cube = world.spawn(BodyDesc::cuboid([0.5, 0.5, 0.5]).position([0.2, 0.5, -0.3]));
    settle(&mut world, 30);
    let resting = features_of(&mut world, cube, floor);
    assert!(
        !resting.is_empty(),
        "a box resting on a triangulated floor must hold a contact point"
    );
    for frame in 0..20usize {
        world.step(DT);
        world.wait();
        assert_eq!(
            features_of(&mut world, cube, floor),
            resting,
            "triangle features must survive resting at frame {frame}"
        );
    }
}

#[test]
fn manifold_points_hold_distinct_features() {
    let mut world = new_world(gravity_config());
    let _floor = wide_floor(&mut world, 0.8);
    let (vertices, faces) = cube_mesh();
    let hull = world.add_hull_from_mesh(&vertices, &faces);
    let bodies = [
        ([0.0, 0.5, 0.0], BodyDesc::cuboid([0.5, 0.5, 0.5])),
        (
            [1.6, 0.5, 0.0],
            BodyDesc::new(ColliderDesc::new(Shape::hull(hull))),
        ),
        (
            [-1.6, 0.75, 0.0],
            BodyDesc::new(ColliderDesc::new(Shape::cylinder(0.5, 0.5))),
        ),
        (
            [0.0, 0.75, 1.6],
            BodyDesc::new(ColliderDesc::new(Shape::capsule(0.4, 0.4))),
        ),
        ([3.2, 2.5, 0.0], BodyDesc::sphere(0.5)),
    ];
    for (position, desc) in bodies {
        world.spawn(desc.position(position).friction(0.8));
    }
    for frame in 0..120usize {
        world.step(DT);
        if !frame.is_multiple_of(10) {
            continue;
        }
        world.wait();
        for manifold in world.contact_manifolds() {
            let held = manifold
                .points
                .iter()
                .map(|point| point.feature)
                .collect::<Vec<_>>();
            let mut distinct = held.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(
                distinct.len(),
                held.len(),
                "a manifold must identify every point separately at frame {frame}: {held:?}"
            );
        }
    }
}

#[test]
fn sparse_iteration_stack_holds_through_contact_identity() {
    let mut world = new_world(dynamis_model::PhysicsConfig {
        solve_iterations: 2,
        position_iterations: 4,
        ..gravity_config()
    });
    let _floor = wide_floor(&mut world, 0.7);
    let mut boxes = Vec::new();
    for level in 0..6 {
        boxes.push(
            world.spawn(
                BodyDesc::cuboid([0.5, 0.5, 0.5])
                    .position([0.0, 0.5 + level as f32 * 1.005, 0.0])
                    .friction(0.7),
            ),
        );
    }
    settle(&mut world, 300);
    let top = world.read_state(*boxes.last().expect("the stack is not empty"));
    assert!(
        (top.position[1] - 5.5).abs() < 0.15,
        "a stack must hold its spacing on sparse iterations, top at {}",
        top.position[1]
    );
    for cube in &boxes {
        let state = world.read_state(*cube);
        assert!(
            state.position[0].abs() < 0.1 && state.position[2].abs() < 0.1,
            "the stack must stay aligned, got {:?}",
            state.position
        );
    }
}
