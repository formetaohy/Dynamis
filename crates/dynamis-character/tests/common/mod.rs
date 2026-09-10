use dynamis_gpu::{GpuContext, GpuRequest};
use dynamis_model::PhysicsConfig;
use dynamis_simulate::Simulation;
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

pub fn sim(capacity: usize, config: PhysicsConfig) -> Simulation {
    Simulation::new(gpu(), capacity, config)
}

pub fn gravity_config() -> PhysicsConfig {
    PhysicsConfig {
        damping: 0.0,
        angular_damping: 0.0,
        ..PhysicsConfig::default()
    }
}
