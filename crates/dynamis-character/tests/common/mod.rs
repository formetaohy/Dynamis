use dynamis_gpu::{GpuContext, GpuRequest, WarmupBudget};
use dynamis_model::PhysicsConfig;
use dynamis_world::World;
use std::sync::OnceLock;

pub const DT: f32 = 1.0 / 60.0;

static GPU: OnceLock<GpuContext> = OnceLock::new();

pub fn gpu() -> GpuContext {
    GPU.get_or_init(|| {
        pollster::block_on(async {
            GpuContext::open(&GpuRequest::default())
                .await
                .expect("test gpu")
        })
    })
    .clone()
}

pub fn new_world(config: PhysicsConfig) -> World {
    let world = World::new(gpu(), config);
    world.warmup(WarmupBudget::All);
    world
}

pub fn gravity_config() -> PhysicsConfig {
    PhysicsConfig {
        damping: 0.0,
        angular_damping: 0.0,
        ..PhysicsConfig::default()
    }
}
