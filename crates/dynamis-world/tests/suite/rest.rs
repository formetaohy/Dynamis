use super::common::{DT, asleep, distance, gravity_config, new_world, settle, settle_until};
use dynamis_abi::{COUNTER_SOFT_ACTIVE, COUNTER_SOFT_SLEPT, COUNTER_SOFT_WOKE};
use dynamis_model::{BodyDesc, SoftBodyDesc, SoftMaterial};
use dynamis_world::World;

fn ground(world: &mut World) {
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
}

fn cloth(position: [f32; 3]) -> SoftBodyDesc {
    let mut cloth = SoftBodyDesc::cloth([6, 6], 0.25, SoftMaterial::rigid());
    cloth.radius = 0.05;
    cloth.position = position;
    cloth
}

fn soft_asleep(world: &World) -> bool {
    world.measured()[COUNTER_SOFT_ACTIVE] == 0
}

fn quiet(world: &mut World) {
    settle_until(world, 240, |world| soft_asleep(world));
}

fn mean_height(positions: &[[f32; 3]]) -> f32 {
    positions.iter().map(|position| position[1]).sum::<f32>() / positions.len() as f32
}

#[test]
fn a_settled_soft_body_leaves_the_simulation_domain() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let _cloth = world.add_soft_body(cloth([0.0, 0.6, 0.0]));
    settle_until(&mut world, 240, |world| asleep(world) && soft_asleep(world));
    assert!(
        soft_asleep(&world),
        "a settled soft body must report no active soft work"
    );
    assert!(
        world.is_idle(),
        "a world whose only deformable has settled must leave the simulation domain"
    );
}

#[test]
fn a_falling_soft_body_keeps_the_simulation_domain_awake() {
    let mut world = new_world(gravity_config());
    let handle = world.add_soft_body(cloth([0.0, 8.0, 0.0]));
    let before = world.soft_body_positions(handle);
    settle(&mut world, 30);
    assert!(
        world.measured()[COUNTER_SOFT_ACTIVE] > 0,
        "a free falling soft body must stay in the simulation domain"
    );
    assert!(
        !world.is_idle(),
        "a moving soft body must keep the world busy"
    );
    let after = world.soft_body_positions(handle);
    assert!(
        mean_height(&after) < mean_height(&before) - 1.0,
        "a free soft body must fall, got {:?} from {:?}",
        mean_height(&after),
        mean_height(&before)
    );
}

#[test]
fn a_sleeping_soft_body_holds_the_pose_it_slept_at() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let handle = world.add_soft_body(cloth([0.0, 0.6, 0.0]));
    settle_until(&mut world, 240, |world| asleep(world) && soft_asleep(world));
    world.wait();
    let slept = world.soft_body_positions(handle);
    settle(&mut world, 90);
    let after = world.soft_body_positions(handle);
    assert_eq!(
        slept, after,
        "a sleeping soft body must not move until something wakes it"
    );
}

#[test]
fn a_moving_rigid_body_wakes_the_sleeping_soft_body_it_meets() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let _cloth = world.add_soft_body(cloth([0.0, 0.6, 0.0]));
    quiet(&mut world);
    let ball = world.spawn(BodyDesc::sphere(0.3).position([0.25, 4.0, 0.25]));
    settle_until(&mut world, 240, |world| !soft_asleep(world));
    settle(&mut world, 120);
    let resting_height = world.read_state(ball).position[1];
    assert!(
        resting_height > 0.3,
        "the cloth must catch the ball it woke, it fell through to {resting_height}"
    );
}

#[test]
fn a_moved_platform_wakes_the_sleeping_soft_body_it_holds() {
    let mut world = new_world(gravity_config());
    let platform = world.spawn(
        BodyDesc::cuboid([2.0, 0.1, 2.0])
            .position([0.0, 0.1, 0.0])
            .kinematic(true),
    );
    let _cloth = world.add_soft_body(cloth([-0.5, 0.35, -0.5]));
    quiet(&mut world);
    world.set_velocity(platform, [1.0, 0.0, 0.0]);
    let mut woken = 0;
    for _ in 0..60 {
        world.step(DT);
        world.wait();
        woken += world.measured()[COUNTER_SOFT_WOKE];
    }
    assert!(
        woken > 0,
        "a driven platform must wake the sleeping soft body it holds"
    );
}

#[test]
fn a_global_parameter_change_wakes_the_sleeping_simulation() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let ball = world.spawn(BodyDesc::sphere(0.3).position([3.0, 0.3, 0.0]));
    let handle = world.add_soft_body(cloth([0.0, 0.6, 0.0]));
    settle_until(&mut world, 240, |world| asleep(world) && soft_asleep(world));
    assert!(world.read_state(ball).sleeping, "the ball must sleep");
    let resting = mean_height(&world.soft_body_positions(handle));
    world.set_gravity([0.0, 9.81, 0.0]);
    settle(&mut world, 60);
    assert!(
        !world.read_state(ball).sleeping,
        "a gravity change must wake a sleeping rigid body"
    );
    assert!(
        world.read_state(ball).position[1] > 0.4,
        "inverted gravity must lift the sleeping ball, got {:?}",
        world.read_state(ball).position
    );
    let lifted = mean_height(&world.soft_body_positions(handle));
    assert!(
        lifted > resting + 0.3,
        "inverted gravity must lift the sleeping soft body, {resting} -> {lifted}"
    );
}

#[test]
fn a_settled_soft_body_reports_one_sleep_transition() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let _cloth = world.add_soft_body(cloth([0.0, 0.6, 0.0]));
    let mut slept = 0;
    let mut woke = 0;
    for _ in 0..300 {
        world.step(DT);
        world.wait();
        slept += world.measured()[COUNTER_SOFT_SLEPT];
        woke += world.measured()[COUNTER_SOFT_WOKE];
    }
    assert_eq!(slept, 1, "a settling soft body must sleep exactly once");
    assert_eq!(woke, 0, "nothing must wake a settled soft body");
}

#[test]
fn a_restored_world_keeps_the_soft_body_asleep() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let handle = world.add_soft_body(cloth([0.0, 0.6, 0.0]));
    quiet(&mut world);
    let slept = world.soft_body_positions(handle);
    let snapshot = world.snapshot();

    let mut restored = new_world(gravity_config());
    restored.restore(&snapshot);
    settle(&mut restored, 90);
    assert!(
        soft_asleep(&restored),
        "a restored world must keep its soft bodies asleep"
    );
    assert!(
        restored.is_idle(),
        "a restored settled world must leave the simulation domain"
    );
    let after = restored.soft_body_positions(handle);
    assert!(
        distance(after[0], slept[0]) < 1e-6,
        "a restored sleeping soft body must hold its pose"
    );
}
