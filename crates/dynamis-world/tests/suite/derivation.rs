use super::common::{DT, asleep, gravity_config, observed_world, settle, settle_until};
use dynamis_abi::{COUNTER_IMMOVABLE_EMITTED, COUNTER_PAIRS, COUNTER_RESTING_REBUILD};
use dynamis_model::{BodyDesc, ColliderDesc, Shape, SoftBodyDesc};

fn ground(world: &mut dynamis_world::World, half_x: f32) {
    world.spawn(
        BodyDesc::cuboid([half_x, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
}

fn resting_blocks(
    world: &mut dynamis_world::World,
    count: usize,
) -> Vec<dynamis_model::BodyHandle> {
    (0..count)
        .map(|index| {
            world.spawn(BodyDesc::cuboid([0.4; 3]).position([
                (index % 8) as f32 - 3.5,
                0.5,
                (index / 8) as f32 - 2.5,
            ]))
        })
        .collect()
}

fn hold_quiet(world: &mut dynamis_world::World, frames: usize) {
    for _ in 0..frames {
        world.step(DT);
    }
    world.wait();
}

#[test]
fn a_released_capacity_plan_still_holds_the_immovable_grid() {
    let mut world = observed_world(gravity_config());
    ground(&mut world, 20.0);
    let blocks = resting_blocks(&mut world, 48);
    settle_until(&mut world, 900, |world| asleep(world));
    hold_quiet(&mut world, 240);
    let probe = blocks[0];
    world.wake(probe);
    for _ in 0..180 {
        world.apply_force(probe, [0.0, -2000.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let height = world.read_state(probe).position[1];
    assert!(
        height > 0.0,
        "a body driven long after the plan released its capacity must still meet the immovable floor, got {height}"
    );
    assert!(
        world.measured()[COUNTER_PAIRS] > 0,
        "the driven body must pair with the immovable floor"
    );
}

#[test]
fn a_movable_step_leaves_the_immovable_grid_alone() {
    let mut world = observed_world(gravity_config());
    ground(&mut world, 20.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 6.0, 0.0]));
    world.step(DT);
    world.wait();
    for _ in 0..12 {
        world.step(DT);
        world.wait();
        assert_eq!(
            world.measured()[COUNTER_IMMOVABLE_EMITTED],
            0,
            "a step that only moves a body must not derive the immovable entries"
        );
    }
    assert!(world.read_state(ball).position[1] < 6.0);
}

#[test]
fn an_immovable_edit_derives_the_immovable_grid() {
    let mut world = observed_world(gravity_config());
    ground(&mut world, 20.0);
    settle_until(&mut world, 240, |world| asleep(world));
    let platform = world.spawn(
        BodyDesc::cuboid([2.0, 0.5, 2.0])
            .mass(0.0)
            .position([6.0, 0.5, 0.0]),
    );
    world.step(DT);
    world.wait();
    assert!(
        world.measured()[COUNTER_IMMOVABLE_EMITTED] > 0,
        "adding an immovable collider must derive the immovable entries"
    );
    let ball = world.spawn(BodyDesc::sphere(0.5).position([6.0, 4.0, 0.0]));
    settle_until(&mut world, 600, |world| {
        (world.read_state(ball).position[1] - 1.5).abs() < 0.2
    });
    world.set_position(platform, [11.0, 0.5, 0.0]);
    world.wake(ball);
    world.set_position(ball, [11.0, 4.0, 0.0]);
    settle_until(&mut world, 600, |world| {
        (world.read_state(ball).position[1] - 1.5).abs() < 0.2
    });
}

#[test]
fn an_edited_resting_collider_derives_the_resting_grid() {
    let mut world = observed_world(gravity_config());
    ground(&mut world, 20.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.5, 0.0]));
    settle_until(&mut world, 600, |world| asleep(world));
    world.set_collider(ball, 0, ColliderDesc::new(Shape::sphere(1.0)));
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_RESTING_REBUILD],
        1,
        "editing the collider of a resting body must derive the resting entries again"
    );
}

#[test]
fn an_edited_sleeping_body_derives_the_resting_grid() {
    let mut world = observed_world(gravity_config());
    ground(&mut world, 20.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.5, 0.0]));
    settle_until(&mut world, 600, |world| asleep(world));
    world.set_position(ball, [3.0, 0.5, 0.0]);
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_RESTING_REBUILD],
        1,
        "editing a resting body must derive the resting entries on the step that declares the edit"
    );
}

#[test]
fn a_restored_scene_derives_every_grid_again() {
    let mut world = observed_world(gravity_config());
    ground(&mut world, 20.0);
    world.spawn(BodyDesc::cuboid([0.4; 3]).position([2.0, 0.5, 0.0]));
    settle_until(&mut world, 600, |world| asleep(world));
    let snapshot = world.snapshot();
    world.spawn(BodyDesc::cuboid([0.4; 3]).position([4.0, 0.5, 0.0]));
    world.step(DT);
    world.wait();
    world.restore(&snapshot);
    world.step(DT);
    world.wait();
    assert!(
        world.measured()[COUNTER_IMMOVABLE_EMITTED] > 0,
        "a restored scene must derive the immovable entries again"
    );
    assert_eq!(
        world.measured()[COUNTER_RESTING_REBUILD],
        1,
        "a restored scene must derive the resting entries again"
    );
}

#[test]
fn a_particle_radius_edit_derives_the_grids_at_the_resolution_it_moves() {
    let mut world = observed_world(gravity_config());
    ground(&mut world, 50.0);
    let soft = world.add_soft_body(SoftBodyDesc::new(vec![[500.0, 0.0, 0.0]], vec![]).radius(0.5));
    settle(&mut world, 4);
    let ball = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 10.0, 0.0])
            .velocity([0.0, -20.0, 0.0]),
    );
    settle(&mut world, 1);
    world.set_soft_particle_radius(soft, 0, 0.01);
    world.step(DT);
    world.wait();
    assert!(
        world.measured()[COUNTER_IMMOVABLE_EMITTED] > 0,
        "a particle radius that moves the grid resolution must derive the immovable entries"
    );
    settle(&mut world, 40);
    let height = world.read_state(ball).position[1];
    assert!(
        (height - 0.5).abs() < 0.2,
        "a body must still meet an immovable floor keyed at the resolution a particle radius edit moved, got {height}"
    );
}

#[test]
fn a_material_edit_leaves_both_grids_alone() {
    let mut world = observed_world(gravity_config());
    ground(&mut world, 20.0);
    let block = world.spawn(BodyDesc::cuboid([0.4; 3]).position([2.0, 0.5, 0.0]));
    settle_until(&mut world, 600, |world| asleep(world));
    world.set_friction(block, 0.9);
    world.set_restitution(block, 0.1);
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_IMMOVABLE_EMITTED],
        0,
        "a material edit moves neither a cell nor the resolution the immovable entries are keyed at"
    );
    assert_eq!(
        world.measured()[COUNTER_RESTING_REBUILD],
        0,
        "a material edit moves neither a cell nor the resolution the resting entries are keyed at"
    );
}

#[test]
fn a_static_material_edit_leaves_both_grids_alone() {
    let mut world = observed_world(gravity_config());
    let floor = world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    world.spawn(BodyDesc::cuboid([0.4; 3]).position([2.0, 0.5, 0.0]));
    settle_until(&mut world, 600, |world| asleep(world));
    world.set_friction(floor, 0.9);
    world.set_collider_events(floor, 0, dynamis_model::ContactEventMode::Persist);
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_IMMOVABLE_EMITTED],
        0,
        "an immovable collider's materials are read live, so they owe the grid no derivation"
    );
    assert_eq!(
        world.measured()[COUNTER_RESTING_REBUILD],
        0,
        "an immovable collider's materials cannot move a resting body's entries"
    );
}

#[test]
fn an_edited_movable_collider_size_derives_the_immovable_grid() {
    let mut world = observed_world(gravity_config());
    ground(&mut world, 20.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 8.0, 0.0]));
    settle(&mut world, 1);
    world.set_collider(ball, 0, ColliderDesc::new(Shape::sphere(0.001)));
    world.step(DT);
    world.wait();
    assert!(
        world.measured()[COUNTER_IMMOVABLE_EMITTED] > 0,
        "a movable collider's size is a maximum the grid resolution is derived from, so the immovable entries must be keyed at the resolution it moves"
    );
}

#[test]
fn a_particle_radius_edit_derives_the_resting_grid_at_the_resolution_it_moves() {
    let mut world = observed_world(gravity_config());
    ground(&mut world, 50.0);
    let sleeper = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.5, 0.0]));
    let soft = world.add_soft_body(SoftBodyDesc::new(vec![[500.0, 0.0, 0.0]], vec![]).radius(0.5));
    settle_until(&mut world, 600, |world| asleep(world));
    world.set_soft_particle_radius(soft, 0, 0.01);
    world.step(DT);
    world.wait();
    assert!(
        world.measured()[COUNTER_RESTING_REBUILD] == 1,
        "a particle radius that moves the grid resolution must derive the resting entries"
    );
    let faller = world.spawn(BodyDesc::sphere(0.5).position([0.0, 3.0, 0.0]));
    settle(&mut world, 90);
    let rested = world.read_state(faller).position[1];
    assert!(
        rested > 1.0,
        "a body must still meet a sleeping body's entries keyed at the resolution a particle radius edit moved, got {rested}"
    );
    assert!(world.read_state(sleeper).position[1] > 0.0);
}
