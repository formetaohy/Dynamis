use super::common::DT;
use dynamis_gpu::{GpuContext, GpuRequest, WarmupBudget};
use dynamis_model::{BodyDesc, PhysicsConfig};
use dynamis_world::World;
use std::time::Duration;

fn cold_world() -> World {
    let context = pollster::block_on(GpuContext::open(&GpuRequest::default())).expect("test gpu");
    let mut world = World::new(context, PhysicsConfig::default());
    world.observe_all_bodies();
    world
}

fn falling_scene(world: &mut World) {
    world.spawn(BodyDesc::sphere(0.5).position([0.0, 2.0, 0.0]));
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
}

#[test]
fn a_cold_world_declares_every_kernel_without_compiling_one() {
    let world = cold_world();
    assert!(
        !world.is_warm(),
        "constructing a world must not compile its pipelines"
    );
    let progress = world.warmup(WarmupBudget::Within(Duration::ZERO));
    assert_eq!(progress.ready, 1, "a zero budget compiles one kernel");
    assert!(
        progress.total > progress.ready,
        "a world declares more than one kernel"
    );
}

#[test]
fn a_step_compiles_only_the_kernels_its_frame_dispatches() {
    let mut world = cold_world();
    let declared = world.warmup(WarmupBudget::Within(Duration::ZERO)).total;
    falling_scene(&mut world);
    for _ in 0..8 {
        world.step(DT);
    }
    world.wait();
    let dispatched = world.warmup(WarmupBudget::Within(Duration::ZERO));
    assert_eq!(dispatched.total, declared);
    assert!(
        dispatched.ready > 1,
        "a step compiles the kernels its frame dispatches"
    );
    assert!(
        !dispatched.complete(),
        "a scene that never queries, holds no soft body and enables no ccd must leave those kernels cold"
    );
    assert!(
        world.warmup(WarmupBudget::All).complete(),
        "an explicit warmup still compiles the whole graph"
    );
}

#[test]
fn a_step_needs_no_warmup() {
    let mut world = cold_world();
    falling_scene(&mut world);
    for _ in 0..8 {
        world.step(DT);
    }
    world.wait();
    assert!(
        world.read_state(world.bodies()[0]).position[1] < 2.0,
        "a frame simulates without a prior warmup"
    );
    assert!(
        !world.is_warm(),
        "stepping must not compile kernels the scene never reaches"
    );
}
