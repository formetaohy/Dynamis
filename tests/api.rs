use dynamis::{BodyDesc, GpuContext, PhysicsConfig, Simulation};
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

fn static_config() -> PhysicsConfig {
    PhysicsConfig {
        gravity: [0.0, 0.0, 0.0],
        damping: 0.0,
        angular_damping: 0.0,
        ..PhysicsConfig::default()
    }
}

#[test]
fn grow_preserves_bodies_and_state() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let ball = sim.spawn(
        BodyDesc::sphere(0.3).position([0.0, 5.0, 0.0]).velocity([1.0, 0.0, 0.0]),
    );
    for _ in 0..10 {
        sim.step(DT);
    }
    sim.wait();
    let before = sim.read_state(ball).position;
    sim.grow(64);
    for _ in 0..10 {
        sim.step(DT);
    }
    sim.wait();
    let after = sim.read_state(ball).position;
    assert!(after[0] > before[0] + 0.1, "grow must preserve simulation flow");
    assert!(
        (after[1] - before[1]).abs() < 1e-4,
        "gravity-free path must stay flat, got {} vs {}",
        after[1],
        before[1]
    );
}

#[test]
fn grow_then_spawn_uses_free_capacity() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    sim.spawn(BodyDesc::sphere(0.2));
    sim.grow(16);
    let extra = sim.spawn(BodyDesc::sphere(0.2).position([0.0, 4.0, 0.0]));
    sim.step(DT);
    sim.wait();
    let state = sim.read_state(extra);
    assert!(state.position[1] > 0.0, "grown simulation must accept new bodies");
}

#[test]
fn cloned_context_runs_two_worlds() {
    let (_guard, gpu) = serialized_gpu();
    let mut first = Simulation::new(gpu.clone(), 8, static_config());
    let mut second = Simulation::new(gpu, 8, static_config());
    let a = first.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]).velocity([1.0, 0.0, 0.0]));
    let b = second.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]).velocity([2.0, 0.0, 0.0]));
    for _ in 0..10 {
        first.step(DT);
        second.step(DT);
    }
    first.wait();
    second.wait();
    assert!(
        first.read_state(a).position[0] < second.read_state(b).position[0],
        "independent worlds must simulate independently"
    );
}

#[test]
fn force_at_point_spins_body() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let body = sim.spawn(BodyDesc::cuboid([0.5, 0.5, 0.5]));
    sim.apply_force_at_point(body, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    for _ in 0..10 {
        sim.step(DT);
    }
    sim.wait();
    let spin = sim.read_state(body).angular_velocity;
    assert!(
        spin.iter().any(|w| w.abs() > 1e-5),
        "off-center force must spin the body, got {spin:?}"
    );
}

#[test]
fn angular_impulse_spins_body() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let body = sim.spawn(BodyDesc::sphere(0.5));
    sim.apply_angular_impulse(body, [0.0, 0.0, 4.0]);
    sim.step(DT);
    sim.wait();
    let spin = sim.read_state(body).angular_velocity;
    assert!(spin[2] > 0.1, "angular impulse must spin the body, got {spin:?}");
}

#[test]
fn set_gravity_affects_fall() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        8,
        PhysicsConfig {
            gravity: [0.0, -9.81, 0.0],
            damping: 0.0,
            angular_damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let body = sim.spawn(BodyDesc::sphere(0.2).position([0.0, 10.0, 0.0]));
    sim.set_gravity([0.0, -4.0, 0.0]);
    for _ in 0..30 {
        sim.step(DT);
    }
    sim.wait();
    let y = sim.read_state(body).position[1];
    let expected = 10.0 - 0.5 * 4.0 * DT * DT * (30.0 * 31.0);
    assert!(
        (y - expected).abs() < 1e-3,
        "updated gravity must drive the fall, got {y} vs {expected}"
    );
}

#[test]
fn hull_and_mesh_share_shape_sources() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let vertices = vec![[-1.0f32, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
    let triangles = vec![[0u32, 2, 1], [0, 3, 2]];
    let source = sim.add_mesh(&vertices, &triangles);
    sim.spawn(BodyDesc::new(dynamis::ColliderDesc::new(dynamis::Shape::mesh(source))).mass(0.0));
    let ball = sim.spawn(BodyDesc::sphere(0.4).position([0.0, 3.0, 0.0]));
    for _ in 0..90 {
        sim.step(DT);
    }
    sim.wait();
    let y = sim.read_state(ball).position[1];
    assert!(
        (y - 0.4).abs() < 0.05,
        "mesh floor must catch the ball at y=0.4, got {y}"
    );
}
