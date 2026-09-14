use dynamis_abi::COUNTER_ACTIVE;
use dynamis_gpu::{
    AdapterInfo, Backends, Device, GpuContext, GpuRequest, LimitsPolicy, Queue, WarmupBudget,
};
use dynamis_model::{BodyDesc, BodyHandle, ColliderDesc, PhysicsConfig, Shape};
use dynamis_world::World;
use std::sync::OnceLock;

pub const DT: f32 = 1.0 / 60.0;

const SETTLE_POLL: usize = 4;

pub struct Host {
    pub device: Device,
    pub queue: Queue,
    pub info: AdapterInfo,
}

static HOST: OnceLock<Host> = OnceLock::new();
static GPU: OnceLock<GpuContext> = OnceLock::new();

fn request() -> GpuRequest {
    let mut request = GpuRequest::default();
    if let Ok(backend) = std::env::var("DYNAMIS_TEST_BACKEND") {
        request.backends = match backend.as_str() {
            "vulkan" => Backends::VULKAN,
            "dx12" => Backends::DX12,
            _ => request.backends,
        };
    }
    request
}

pub fn host() -> &'static Host {
    HOST.get_or_init(|| {
        let request = request();
        let adapter = pollster::block_on(request.adapter()).expect("test adapter");
        let available = adapter.features();
        let features = request.required_features | (request.optional_features & available);
        let limits = match request.limits {
            LimitsPolicy::Adapter => adapter.limits(),
            LimitsPolicy::Minimum => GpuContext::MINIMUM_LIMITS,
        };
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("dynamis host device"),
            required_features: features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .expect("the host device must be requestable");
        Host {
            device,
            queue,
            info,
        }
    })
}

pub fn gpu() -> GpuContext {
    GPU.get_or_init(|| {
        let host = host();
        GpuContext::adopt(host.device.clone(), host.queue.clone(), host.info.clone())
    })
    .clone()
}

pub fn new_world(config: PhysicsConfig) -> World {
    let world = World::new(gpu(), config);
    world.warmup(WarmupBudget::All);
    world
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

pub fn symplectic_fall(gravity: f32, frames: u32, initial_speed: f32, substeps: u32) -> f32 {
    let step = DT / substeps as f32;
    let count = (frames * substeps) as f32;
    initial_speed * step * count - 0.5 * gravity * step * step * count * (count + 1.0)
}

pub fn settle(world: &mut World, frames: usize) {
    for _ in 0..frames {
        world.step(DT);
    }
    world.wait();
}

pub fn settle_until(
    world: &mut World,
    limit: usize,
    mut settled: impl FnMut(&mut World) -> bool,
) -> usize {
    for frame in 1..=limit {
        world.step(DT);
        if !frame.is_multiple_of(SETTLE_POLL) {
            continue;
        }
        world.wait();
        if settled(world) {
            return frame;
        }
    }
    world.wait();
    assert!(settled(world), "world must settle within {limit} frames");
    limit
}

pub fn asleep(world: &World) -> bool {
    world.measured()[COUNTER_ACTIVE] == 0
}

pub fn converged(previous: &mut f32, sample: f32) -> bool {
    let converged = (*previous - sample).abs() < 1e-4;
    *previous = sample;
    converged
}

pub fn static_sphere_ground(world: &mut World, radius: f32) -> BodyHandle {
    world.spawn(BodyDesc::static_sphere(radius))
}

pub fn flat_mesh_floor(world: &mut World) -> BodyHandle {
    let vertices = vec![
        [-30.0f32, 0.0, -30.0],
        [30.0, 0.0, -30.0],
        [30.0, 0.0, 30.0],
        [-30.0, 0.0, 30.0],
    ];
    let triangles = vec![[0u32, 2, 1], [0, 3, 2]];
    let floor = world.add_mesh(&vertices, &triangles, None);
    world.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(floor))).mass(0.0))
}

pub fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}
