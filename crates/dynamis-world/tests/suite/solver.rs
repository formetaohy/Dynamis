use super::common::{DT, gravity_config, observed_world, settle};
use dynamis_abi::{COUNTER_LIVE, COUNTER_SOLVER_ROWS};
use dynamis_model::{BodyDesc, PhysicsConfig};
use dynamis_world::World;

fn floor(world: &mut World) {
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
}

fn free_fall(world: &mut World, count: usize) {
    for index in 0..count {
        world.spawn(BodyDesc::cuboid([0.5, 0.5, 0.5]).position([index as f32 * 3.0, 60.0, 0.0]));
    }
}

#[test]
fn a_step_that_solves_nothing_claims_no_solver_row() {
    let mut world = observed_world(gravity_config());
    free_fall(&mut world, 32);
    settle(&mut world, 2);
    assert!(
        world.measured()[COUNTER_LIVE] >= 32,
        "bodies in free fall must stay live"
    );
    assert_eq!(
        world.measured()[COUNTER_SOLVER_ROWS],
        0,
        "a step without contacts or constraints must claim no solver row",
    );
}

#[test]
fn a_solve_claims_the_rows_its_blocks_reference_alone() {
    let mut world = observed_world(PhysicsConfig {
        solve_iterations: 4,
        position_iterations: 1,
        ..gravity_config()
    });
    floor(&mut world);
    free_fall(&mut world, 32);
    let resting = world.spawn(
        BodyDesc::cuboid([0.5, 0.5, 0.5])
            .position([0.0, 0.5, 0.0])
            .friction(0.9),
    );
    settle(&mut world, 1);
    assert!(
        world.measured()[COUNTER_LIVE] >= 33,
        "every awake body must stay live, got {}",
        world.measured()[COUNTER_LIVE],
    );
    assert_eq!(
        world.measured()[COUNTER_SOLVER_ROWS],
        2,
        "one contact must claim the resting body and its floor alone, {} live bodies",
        world.measured()[COUNTER_LIVE],
    );

    world.remove(resting);
    settle(&mut world, 2);
    assert_eq!(
        world.measured()[COUNTER_SOLVER_ROWS],
        0,
        "a solver that lost its blocks must claim no row",
    );
}

#[test]
fn a_solve_claims_each_shared_row_once() {
    let mut world = observed_world(gravity_config());
    floor(&mut world);
    for index in 0..4 {
        world.spawn(
            BodyDesc::cuboid([0.5, 0.5, 0.5])
                .position([index as f32 * 1.05 - 1.6, 0.5, 0.0])
                .friction(0.9),
        );
    }
    settle(&mut world, 1);
    assert_eq!(
        world.measured()[COUNTER_SOLVER_ROWS],
        5,
        "four stacked cuboids on one floor must claim five distinct rows",
    );
    settle(&mut world, 240);
    assert_eq!(
        world.measured()[COUNTER_SOLVER_ROWS],
        0,
        "a settled pile must leave the solver without rows",
    );
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_SOLVER_ROWS],
        0,
        "a rest step must not resurrect the rows of its settled pile",
    );
}
