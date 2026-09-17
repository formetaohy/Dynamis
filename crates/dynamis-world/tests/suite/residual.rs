use super::common::{DT, asleep, gravity_config, observed_world, settle, settle_until};
use dynamis_model::{BodyDesc, ColliderDesc, PhysicsConfig, Shape};

fn floor(world: &mut dynamis_world::World, friction: f32) {
    world.spawn(
        BodyDesc::cuboid([10.0, 0.5, 10.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0])
            .friction(friction),
    );
}

fn leaning_tower(world: &mut dynamis_world::World) {
    floor(world, 0.6);
    for index in 0..10 {
        world.spawn(
            BodyDesc::new(ColliderDesc::new(Shape::cuboid([0.5, 0.5, 0.5])).friction(0.6))
                .position([0.0, 0.5 + index as f32, 0.0])
                .velocity([0.6, 0.0, 0.0]),
        );
    }
}

#[test]
fn a_stacked_solve_reports_the_velocity_its_budget_leaves() {
    let mut world = observed_world(gravity_config());
    leaning_tower(&mut world);
    settle(&mut world, 90);
    world.set_config(PhysicsConfig {
        solve_iterations: 1,
        ..gravity_config()
    });
    world.step(DT);
    world.wait();
    let short = world.solve_residual();
    world.set_config(gravity_config());
    world.step(DT);
    world.wait();
    let full = world.solve_residual();
    assert!(
        short.linear > full.linear && short.angular > full.angular,
        "a one iteration budget must leave more velocity than the full budget, \
         {short:?} vs {full:?}",
    );
    assert!(
        full.linear > 0.0 && full.angular > 0.0,
        "a leaning tower must leave its solve moving, got {full:?}",
    );
}

#[test]
fn the_solve_residual_falls_to_nothing_once_the_scene_rests() {
    let mut world = observed_world(gravity_config());
    leaning_tower(&mut world);
    settle_until(&mut world, 900, |world| asleep(world));
    assert_eq!(
        world.solve_residual(),
        dynamis_world::SolveResidual {
            linear: 0.0,
            angular: 0.0,
        },
        "a step whose solve idles must report no residual",
    );
}
