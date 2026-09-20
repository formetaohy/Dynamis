use super::common::{
    DT, asleep, gravity_config, observed_world, settle, settle_until, static_config,
};
use dynamis_abi::COUNTER_SOFT_ACTIVE;
use dynamis_model::{
    BodyDesc, BodyHandle, ColliderDesc, ConstraintDesc, Shape, SoftBodyDesc, SoftMaterial,
    SurfaceTable,
};
use dynamis_world::World;

fn ground(world: &mut World) -> BodyHandle {
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    )
}

fn sleeping_sphere(world: &mut World) -> BodyHandle {
    let body = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.5, 0.0]));
    settle_until(world, 240, |world| asleep(world));
    body
}

fn cloth(position: [f32; 3]) -> SoftBodyDesc {
    let mut cloth = SoftBodyDesc::cloth([4, 4], 0.4, SoftMaterial::rigid());
    cloth.radius = 0.1;
    cloth.position = position;
    cloth
}

fn quiet(world: &mut World) {
    settle_until(world, 480, |world| {
        world.measured()[COUNTER_SOFT_ACTIVE] == 0
    });
}

fn mean_height(world: &mut World, body: dynamis_model::SoftBodyHandle) -> f32 {
    let positions = world.inspect_soft_particles(body);
    positions.iter().map(|position| position[1]).sum::<f32>() / positions.len() as f32
}

#[test]
fn a_moved_static_collider_releases_the_sleeping_body_it_held() {
    let mut world = observed_world(gravity_config());
    let floor = ground(&mut world);
    let body = sleeping_sphere(&mut world);
    let resting = world.read_state(body).position;
    world.set_collider(
        floor,
        0,
        ColliderDesc::new(Shape::cuboid([20.0, 0.5, 20.0])).offset([0.0, -20.0, 0.0]),
    );
    settle(&mut world, 30);
    assert!(
        world.read_state(body).position[1] < resting[1] - 0.5,
        "the collider a sleeping body rests on moved, so the body must fall: {resting:?} -> {:?}",
        world.read_state(body).position,
    );
}

#[test]
fn a_grown_static_collider_lifts_the_sleeping_body_it_reaches() {
    let mut world = observed_world(gravity_config());
    let floor = ground(&mut world);
    let body = sleeping_sphere(&mut world);
    let resting = world.read_state(body).position;
    world.set_collider(
        floor,
        0,
        ColliderDesc::new(Shape::cuboid([20.0, 20.5, 20.0])).offset([0.0, 0.0, 0.0]),
    );
    settle(&mut world, 30);
    assert!(
        world.read_state(body).position[1] > resting[1] + 0.5,
        "the collider a sleeping body rests on grew into it, so the body must be pushed out: \
         {resting:?} -> {:?}",
        world.read_state(body).position,
    );
}

#[test]
fn a_moved_mesh_floor_releases_the_sleeping_body_it_held() {
    let mut world = observed_world(gravity_config());
    let vertices = vec![
        [-20.0f32, 0.0, -20.0],
        [20.0, 0.0, -20.0],
        [20.0, 0.0, 20.0],
        [-20.0, 0.0, 20.0],
    ];
    let triangles = vec![[0u32, 2, 1], [0, 3, 2]];
    let floor = world.add_mesh(&vertices, &triangles, None::<SurfaceTable<'_>>);
    world.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(floor))).mass(0.0));
    let body = sleeping_sphere(&mut world);
    let resting = world.read_state(body).position;
    let lowered = vertices
        .iter()
        .map(|vertex| [vertex[0], vertex[1] - 20.0, vertex[2]])
        .collect::<Vec<_>>();
    world.update_mesh(floor, &lowered, &triangles, None::<SurfaceTable<'_>>);
    settle(&mut world, 30);
    assert!(
        world.read_state(body).position[1] < resting[1] - 0.5,
        "the geometry a sleeping body rests on moved, so the body must fall: {resting:?} -> {:?}",
        world.read_state(body).position,
    );
}

#[test]
fn a_resized_sleeping_body_wakes_itself() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let body = world.spawn(BodyDesc::cuboid([0.5, 0.5, 0.5]).position([0.0, 0.5, 0.0]));
    settle_until(&mut world, 240, |world| asleep(world));
    let resting = world.read_state(body).position;
    world.set_collider(body, 0, ColliderDesc::new(Shape::cuboid([0.5, 1.0, 0.5])));
    settle(&mut world, 30);
    let moved = world.read_state(body).position;
    assert!(
        moved[1] > resting[1] + 0.25,
        "a sleeping body whose collider grew into the floor must wake and be pushed out: \
         {resting:?} -> {moved:?}",
    );
}

#[test]
fn a_removed_body_releases_the_sleeping_body_it_held() {
    let mut world = observed_world(gravity_config());
    let floor = ground(&mut world);
    let body = sleeping_sphere(&mut world);
    let resting = world.read_state(body).position;
    world.remove(floor);
    settle(&mut world, 30);
    assert!(
        world.read_state(body).position[1] < resting[1] - 0.5,
        "the body a sleeping body rested on left the scene, so it must fall: {resting:?} -> {:?}",
        world.read_state(body).position,
    );

    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let lower = world.spawn(BodyDesc::cuboid([1.0, 0.5, 1.0]).position([0.0, 0.5, 0.0]));
    let upper = world.spawn(BodyDesc::cuboid([1.0, 0.5, 1.0]).position([0.0, 1.5, 0.0]));
    settle_until(&mut world, 900, |world| asleep(world));
    let resting = world.read_state(upper).position;
    world.remove(lower);
    settle(&mut world, 30);
    assert!(
        world.read_state(upper).position[1] < resting[1] - 0.5,
        "the support a sleeping body rested on left the scene, so it must fall: {resting:?} -> {:?}",
        world.read_state(upper).position,
    );
}

#[test]
fn a_motor_drives_a_sleeping_joint() {
    let mut world = observed_world(static_config());
    let anchor = world.spawn(BodyDesc::static_sphere(0.2));
    let arm = world.spawn(BodyDesc::cuboid([1.0, 0.1, 0.1]).position([1.0, 0.0, 0.0]));
    let joint = world.add_constraint(
        anchor,
        arm,
        ConstraintDesc::revolute([0.0; 3], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    );
    settle_until(&mut world, 240, |world| asleep(world));
    let resting = world.read_state(arm).position;
    world.set_motor(joint, 2.0, 100.0);
    settle(&mut world, 30);
    let moved = world.read_state(arm).position;
    assert!(
        (moved[1] - resting[1]).abs() > 0.1 || (moved[2] - resting[2]).abs() > 0.1,
        "a motor declared on a sleeping joint must wake and turn the arm: {resting:?} -> {moved:?}",
    );
}

#[test]
fn a_joint_declared_between_sleeping_bodies_wakes_them() {
    let mut world = observed_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.25));
    let second = world.spawn(BodyDesc::sphere(0.25).position([2.0, 0.0, 0.0]));
    settle_until(&mut world, 240, |world| asleep(world));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0),
    );
    settle(&mut world, 60);
    let a = world.read_state(first).position;
    let b = world.read_state(second).position;
    let separation = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    assert!(
        (separation - 1.0).abs() < 0.1,
        "a joint declared between sleeping bodies must wake and pull them together, got {separation}",
    );
}

#[test]
fn a_removed_joint_releases_the_sleeping_body_it_held() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let anchor = world.spawn(BodyDesc::static_sphere(0.2).position([0.0, 4.0, 0.0]));
    let body = world.spawn(BodyDesc::sphere(0.25).position([0.0, 1.0, 0.0]));
    let joint = world.add_constraint(
        anchor,
        body,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 3.0),
    );
    settle_until(&mut world, 480, |world| asleep(world));
    let resting = world.read_state(body).position;
    world.remove_constraint(joint);
    settle(&mut world, 60);
    assert!(
        world.read_state(body).position[1] < resting[1] - 0.5,
        "the joint a sleeping body hung from left the scene, so it must fall: {resting:?} -> {:?}",
        world.read_state(body).position,
    );
}

#[test]
fn a_refreshed_record_leaves_the_sleeping_body_asleep() {
    let mut world = observed_world(gravity_config());
    let floor = ground(&mut world);
    let body = sleeping_sphere(&mut world);
    let resting = world.read_state(body).position;
    world.set_collider(
        floor,
        0,
        ColliderDesc::new(Shape::cuboid([20.0, 0.5, 20.0])),
    );
    world.set_friction(floor, 0.9);
    world.set_restitution(floor, 0.1);
    settle(&mut world, 12);
    assert_eq!(
        world.read_state(body).position,
        resting,
        "a record replaced with the one the device holds must leave the sleeping body alone",
    );
    assert!(
        world.read_state(body).sleeping,
        "a record replaced with the one the device holds must not wake the sleeping body",
    );
}

#[test]
fn a_moved_static_collider_releases_the_sleeping_cloth_it_held() {
    let mut world = observed_world(gravity_config());
    let floor = ground(&mut world);
    let body = world.add_soft_body(cloth([0.0, 1.5, 0.0]));
    quiet(&mut world);
    let resting = mean_height(&mut world, body);
    world.set_collider(
        floor,
        0,
        ColliderDesc::new(Shape::cuboid([20.0, 0.5, 20.0])).offset([0.0, -20.0, 0.0]),
    );
    settle(&mut world, 60);
    assert!(
        mean_height(&mut world, body) < resting - 0.5,
        "the collider a sleeping cloth rests on moved, so the cloth must fall: {resting} -> {}",
        mean_height(&mut world, body),
    );
}

#[test]
fn a_removed_body_releases_the_sleeping_cloth_it_held() {
    let mut world = observed_world(gravity_config());
    let floor = ground(&mut world);
    let body = world.add_soft_body(cloth([0.0, 1.5, 0.0]));
    quiet(&mut world);
    let resting = mean_height(&mut world, body);
    world.remove(floor);
    settle(&mut world, 60);
    assert!(
        mean_height(&mut world, body) < resting - 0.5,
        "the body a sleeping cloth rests on left the scene, so the cloth must fall: {resting} -> {}",
        mean_height(&mut world, body),
    );
}

#[test]
fn a_spawned_static_body_wakes_the_sleeping_cloth_it_reaches() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let _cloth = world.add_soft_body(cloth([0.0, 1.5, 0.0]));
    quiet(&mut world);
    world.spawn(
        BodyDesc::cuboid([0.5; 3])
            .mass(0.0)
            .position([0.4, 0.3, 0.4]),
    );
    let mut woke = 0;
    for _ in 0..30 {
        world.step(DT);
        world.wait();
        woke += world.measured()[dynamis_abi::COUNTER_SOFT_WOKE];
    }
    assert!(
        woke > 0,
        "a spawned body the sleeping cloth answers must wake it",
    );
}
