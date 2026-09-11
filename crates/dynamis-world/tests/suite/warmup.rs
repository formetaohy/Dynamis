use super::common::DT;
use dynamis_gpu::{GpuContext, GpuRequest, WarmupBudget};
use dynamis_model::PhysicsConfig;
use dynamis_world::World;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Duration;

#[test]
fn a_cold_simulation_reports_progress_and_refuses_to_step() {
    let context = pollster::block_on(GpuContext::open(&GpuRequest::default())).expect("test gpu");
    let mut world = World::new(context, PhysicsConfig::default());
    assert!(
        !world.is_warm(),
        "constructing a world must not compile its pipelines"
    );

    let progress = world.warmup(WarmupBudget::Within(Duration::ZERO));
    assert_eq!(progress.ready, 1);
    assert!(
        progress.total > 1,
        "a world declares more than one pipeline"
    );
    assert!(!progress.complete());

    let refused = catch_unwind(AssertUnwindSafe(|| world.step(DT)));
    assert!(
        refused.is_err(),
        "a step before warmup must expose the cold pipeline"
    );
}
