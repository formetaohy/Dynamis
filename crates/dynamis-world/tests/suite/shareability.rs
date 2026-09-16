use super::common::{DT, gravity_config, observed_world};
use dynamis_model::BodyDesc;
use dynamis_world::World;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn the_engine_moves_between_threads_and_serves_shared_reads() {
    assert_send::<World>();
    assert_sync::<World>();

    let mut world = observed_world(gravity_config());
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.0, 0.0]));
    for _ in 0..4 {
        world.step(DT);
    }
    let mut world = std::thread::spawn(move || {
        for _ in 0..12 {
            world.step(DT);
        }
        world
    })
    .join()
    .expect("a worker thread owns and steps the engine");
    world.wait();
    let height = world.read_state(ball).position[1];
    assert!(
        height < 1.0,
        "the worker thread must advance the bodies it owns, got {height}"
    );

    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                assert_eq!(world.count(), 1);
                assert_eq!(world.config().gravity, [0.0, -9.81, 0.0]);
                assert!(world.state_buffer().size() > 0);
                assert!(!world.pass_labels().is_empty());
            });
        }
    });
}
