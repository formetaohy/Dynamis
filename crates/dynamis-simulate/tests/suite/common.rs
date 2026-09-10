use dynamis_gpu::{Backends, GpuContext, GpuRequest};
use dynamis_model::{BodyDesc, BodyHandle, ColliderDesc, PhysicsConfig, Shape};
use dynamis_simulate::Simulation;
use std::sync::OnceLock;

pub const DT: f32 = 1.0 / 60.0;

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

pub fn sim(capacity: usize, config: PhysicsConfig) -> Simulation {
    Simulation::new(gpu(), capacity, config)
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
