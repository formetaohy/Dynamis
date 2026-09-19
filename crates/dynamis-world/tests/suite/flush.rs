use super::common::{DT, gravity_config, observed_world};
use dynamis_model::{BodyDesc, QueryFilter};

#[test]
fn one_run_hands_over_every_stage_of_records_it_composed() {
    let mut world = observed_world(gravity_config());
    let ball = world.spawn(BodyDesc::sphere(0.2).position([0.0, 5.0, 0.0]));
    world.apply_impulse(ball, [1.0, 0.0, 0.0]);
    let query = world.ray_query(
        [-5.0, 5.0, 0.0],
        [1.0, 0.0, 0.0],
        10.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let state = world.read_state(ball);
    assert!(
        state.position[1] < 5.0,
        "the record a scene change moved must reach the device with the step that integrates it, got {:?}",
        state.position
    );
    assert!(
        state.velocity[0] >= 1.0,
        "the record a command composed must reach the device with the step that applies it, got {:?}",
        state.velocity
    );
    assert!(
        world.query_hit(query).is_some(),
        "the record a run registered must reach the device with the run that casts it"
    );
}
