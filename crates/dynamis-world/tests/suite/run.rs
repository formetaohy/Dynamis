use super::common::{DT, new_world, settle, static_config};
use dynamis_model::{BodyDesc, QueryFilter};

#[test]
fn a_query_run_sees_a_body_spawned_since_the_last_step() {
    let mut world = new_world(static_config());
    settle(&mut world, 1);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 2.0, 0.0]));
    let query = world.ray_query(
        [0.0, 2.0, -5.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    world.wait();
    let hit = world
        .query_hit(query)
        .expect("a query run must upload every body the host spawned since the last step");
    assert_eq!(hit.body(), ball);
}

#[test]
fn a_publish_run_leaves_pending_step_inputs_for_the_step() {
    let mut world = new_world(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 4.0, 0.0]));
    settle(&mut world, 2);
    let before = world.read_state(ball).velocity;
    world.step(DT);
    world.apply_force(ball, [0.0, 600.0, 0.0]);
    world.wait();
    world.step(DT);
    world.wait();
    let after = world.read_state(ball).velocity;
    let rise = after[1] - before[1];
    assert!(
        (rise - 600.0 * DT).abs() < 1e-2,
        "a force pending through a publication must land exactly once in the next step, rose {rise}"
    );
}
