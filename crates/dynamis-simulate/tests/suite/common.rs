use dynamis_gpu::{Backends, GpuContext, GpuRequest, WarmupBudget};
use dynamis_layout::COUNTER_ACTIVE;
use dynamis_model::{BodyDesc, BodyHandle, ColliderDesc, PhysicsConfig, Shape};
use dynamis_simulate::Simulation;
use std::sync::OnceLock;

pub const DT: f32 = 1.0 / 60.0;

const SETTLE_POLL: usize = 4;

static GPU: OnceLock<GpuContext> = OnceLock::new();

pub fn gpu() -> GpuContext {
    GPU.get_or_init(|| {
        pollster::block_on(async {
            let mut request = GpuRequest::default();
            if let Ok(backend) = std::env::var("DYNAMIS_TEST_BACKEND") {
                request.backends = match backend.as_str() {
                    "vulkan" => Backends::VULKAN,
                    "dx12" => Backends::DX12,
                    _ => request.backends,
                };
            }
            GpuContext::open(&request).await.expect("test gpu")
        })
    })
    .clone()
}

pub fn sim(config: PhysicsConfig) -> Simulation {
    let simulation = Simulation::new(gpu(), config);
    simulation.warmup(WarmupBudget::All);
    simulation
}

pub fn static_config() -> PhysicsConfig {
    PhysicsConfig {
        gravity: [0.0, 0.0, 0.0],
        damping: 0.0,
        angular_damping: 0.0,
        ..PhysicsConfig::default()
    }
}

pub fn gravity_config() -> PhysicsConfig {
    PhysicsConfig {
        damping: 0.0,
        angular_damping: 0.0,
        ..PhysicsConfig::default()
    }
}

pub fn settle(sim: &mut Simulation, frames: usize) {
    for _ in 0..frames {
        sim.step(DT);
    }
    sim.wait();
}

pub fn settle_until(
    sim: &mut Simulation,
    limit: usize,
    mut settled: impl FnMut(&mut Simulation) -> bool,
) -> usize {
    for frame in 1..=limit {
        sim.step(DT);
        if !frame.is_multiple_of(SETTLE_POLL) {
            continue;
        }
        sim.wait();
        if settled(sim) {
            return frame;
        }
    }
    sim.wait();
    assert!(settled(sim), "world must settle within {limit} frames");
    limit
}

pub fn asleep(sim: &Simulation) -> bool {
    sim.measured()[COUNTER_ACTIVE] == 0
}

pub fn converged(previous: &mut f32, sample: f32) -> bool {
    let converged = (*previous - sample).abs() < 1e-4;
    *previous = sample;
    converged
}

pub fn static_sphere_ground(sim: &mut Simulation, radius: f32) -> BodyHandle {
    sim.spawn(BodyDesc::static_sphere(radius))
}

pub fn flat_mesh_floor(sim: &mut Simulation) -> BodyHandle {
    let vertices = vec![
        [-30.0f32, 0.0, -30.0],
        [30.0, 0.0, -30.0],
        [30.0, 0.0, 30.0],
        [-30.0, 0.0, 30.0],
    ];
    let triangles = vec![[0u32, 2, 1], [0, 3, 2]];
    let floor = sim.add_mesh(&vertices, &triangles);
    sim.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(floor))).mass(0.0))
}

pub fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}
