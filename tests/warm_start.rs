use dynamis::{BodyDesc, GpuContext, PhysicsConfig, Simulation};
use dynamis_layout::ContactRecord;
use std::sync::{Mutex, MutexGuard};

const DT: f32 = 1.0 / 60.0;

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn serialized_gpu() -> (MutexGuard<'static, ()>, GpuContext) {
    let guard = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let context = pollster::block_on(GpuContext::new());
    (guard, context)
}

fn read_records<T: bytemuck::Pod>(sim: &Simulation, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
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

fn read_u32(sim: &Simulation, buffer: &wgpu::Buffer) -> u32 {
    read_records::<u32>(sim, buffer, 3)[0]
}

#[test]
fn rest_contact_carries_impulse_across_steps() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let _ground = sim.spawn(BodyDesc::static_sphere(5.0).position([0.0, -1.0, 0.0]));
    let _ball = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 4.5, 0.0]));
    for _ in 0..60 {
        sim.step(DT);
    }
    sim.wait();
    let count = read_u32(&sim, sim.debug_contact_count());
    assert_eq!(count, 1, "resting ball must hold exactly one contact");
    let contacts: Vec<ContactRecord> = read_records(&sim, sim.debug_contacts(), count as usize);
    let accelerated = contacts[0].points[0].accumulated_normal;
    assert!(
        accelerated > 0.01,
        "resting contact must accumulate normal impulse, got {accelerated}"
    );

    sim.step(DT);
    sim.wait();
    let archived: Vec<ContactRecord> = read_records(
        &sim,
        sim.debug_prev_contacts(),
        count as usize,
    );
    assert_eq!(archived[0].a, contacts[0].a);
    assert_eq!(archived[0].b, contacts[0].b);
    assert_eq!(
        archived[0].points[0].accumulated_normal, accelerated,
        "archived contact must relay the solved impulse into the next frame"
    );
}

#[test]
fn archived_contacts_are_ordered_by_pair() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 16, PhysicsConfig::default());
    let _ground = sim.spawn(BodyDesc::static_sphere(5.0).position([0.0, -1.0, 0.0]));
    let first = sim.spawn(BodyDesc::sphere(0.5).position([-1.0, 4.5, 0.0]));
    let second = sim.spawn(BodyDesc::sphere(0.5).position([1.0, 4.5, 0.0]));
    let third = sim.spawn(BodyDesc::sphere(0.5).position([1.0, 5.5, 0.0]));
    let _ = (first, second, third);
    for _ in 0..60 {
        sim.step(DT);
    }
    sim.wait();
    let count = read_u32(&sim, sim.debug_contact_count());
    assert!(count >= 3, "three resting bodies must hold contacts, got {count}");
    let archived: Vec<ContactRecord> =
        read_records(&sim, sim.debug_prev_contacts(), count as usize);
    for pair in archived.windows(2) {
        assert!(
            (pair[0].a, pair[0].b) <= (pair[1].a, pair[1].b),
            "archived contacts must stay sorted by pair"
        );
    }
}

#[test]
fn fresh_contact_starts_cold_then_warm() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let _ground = sim.spawn(BodyDesc::static_sphere(5.0).position([0.0, -1.0, 0.0]));
    let _ball = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 6.0, 0.0]));
    for _ in 0..70 {
        sim.step(DT);
    }
    sim.wait();
    let count = read_u32(&sim, sim.debug_contact_count());
    assert!(count >= 1, "falling ball must land and hold a contact");
    let contacts: Vec<ContactRecord> = read_records(&sim, sim.debug_contacts(), count as usize);
    let settled = contacts[0].points[0].accumulated_normal;
    assert!(settled > 0.01, "contact must hold a positive impulse");
    for point in &contacts[0].points[..contacts[0].point_count as usize] {
        assert!(
            point.accumulated_normal.is_finite() && point.accumulated_normal >= 0.0,
            "impulse must stay finite and non-negative"
        );
    }
}

#[test]
fn warm_stack_stays_still_after_settling() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 16, PhysicsConfig::default());
    let _ground = sim.spawn(BodyDesc::static_sphere(5.0).position([0.0, -1.0, 0.0]));
    let mut stack = Vec::new();
    for index in 0..3 {
        let body = sim.spawn(
            BodyDesc::sphere(0.4)
                .position([0.0, 4.7 + index as f32 * 0.81, 0.0])
                .restitution(0.0),
        );
        stack.push(body);
    }
    for _ in 0..300 {
        sim.step(DT);
    }
    sim.wait();
    let baseline: Vec<_> = stack
        .iter()
        .map(|body| sim.read_state(*body).position[1])
        .collect();
    for _ in 0..60 {
        sim.step(DT);
    }
    sim.wait();
    for (index, body) in stack.iter().enumerate() {
        let drift = (sim.read_state(*body).position[1] - baseline[index]).abs();
        assert!(
            drift < 1e-3,
            "settled stack body {index} must stay still, drifted {drift}"
        );
    }
}
