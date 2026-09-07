use dynamis_gpu::GpuContext;
use dynamis_model::{BodyDesc, BodyHandle, ColliderDesc, PhysicsConfig, Shape};
use dynamis_simulate::Simulation;
use std::sync::OnceLock;

pub const DT: f32 = 1.0 / 60.0;

static GPU: OnceLock<GpuContext> = OnceLock::new();

pub fn gpu() -> GpuContext {
    GPU.get_or_init(|| pollster::block_on(GpuContext::new()))
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

pub fn read_records<T: bytemuck::Pod>(
    sim: &Simulation,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Vec<T> {
    let bytes = (count * std::mem::size_of::<T>()) as u64;
    let staging = sim.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("record readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = sim
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    sim.queue().submit([encoder.finish()]);
    let _ = sim.device().poll(wgpu::PollType::wait_indefinitely());
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = sim.device().poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map").expect("map error");
    let mapped = slice.get_mapped_range().unwrap();
    bytemuck::cast_slice::<u8, T>(&mapped).to_vec()
}
