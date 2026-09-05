use dynamis_ecs::World;
use dynamis_gpu::GpuContext;
use dynamis_physics::{
    Mass, PhysicsConfig, Restitution, Simulation, SphereCollider, Transform, Velocity,
};

use std::sync::{Mutex, MutexGuard};

const GRAVITY: f32 = 9.81;
const DT: f32 = 1.0 / 60.0;

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn serialized_gpu() -> (MutexGuard<'static, ()>, GpuContext) {
    let guard = GPU_LOCK.lock().expect("gpu lock poisoned");
    let context = pollster::block_on(GpuContext::new());
    (guard, context)
}

fn static_config() -> PhysicsConfig {
    PhysicsConfig {
        gravity: [0.0, 0.0, 0.0],
        damping: 0.0,
        ..PhysicsConfig::default()
    }
}

#[test]
fn free_fall_matches_closed_form() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut world = World::new();
    let mut sim = Simulation::new(
        gpu,
        4,
        PhysicsConfig {
            damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let ball = world.spawn((
        Transform::at([0.0, 10.0, 0.0]),
        Velocity::default(),
        Mass::new(1.0),
        Restitution(0.0),
        SphereCollider::new(0.1),
    ));
    sim.add(ball);

    const STEPS: u32 = 30;
    for _ in 0..STEPS {
        sim.step(&world, DT);
        sim.sync_back(&mut world);
    }
    let expected = 10.0 - 0.5 * GRAVITY * DT * DT * (STEPS as f32 * (STEPS as f32 + 1.0));
    let actual = world.get::<Transform>(ball).unwrap().position[1];
    assert!(
        (actual - expected).abs() < 1e-3,
        "expected {expected}, got {actual}"
    );
    let expected_velocity = -GRAVITY * DT * STEPS as f32;
    let actual_velocity = world.get::<Velocity>(ball).unwrap().linear[1];
    assert!((actual_velocity - expected_velocity).abs() < 1e-3);
}

#[test]
fn overlapping_bodies_separate() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut world = World::new();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let first = world.spawn((
        Transform::at([0.0, -0.4, 0.0]),
        Velocity::default(),
        Mass::new(1.0),
        Restitution(0.0),
        SphereCollider::new(0.5),
    ));
    let second = world.spawn((
        Transform::at([0.0, 0.4, 0.0]),
        Velocity::default(),
        Mass::new(1.0),
        Restitution(0.0),
        SphereCollider::new(0.5),
    ));
    sim.add(first);
    sim.add(second);
    for _ in 0..16 {
        sim.step(&world, DT);
        sim.sync_back(&mut world);
    }
    let first_y = world.get::<Transform>(first).unwrap().position[1];
    let second_y = world.get::<Transform>(second).unwrap().position[1];
    let separation = second_y - first_y;
    assert!(separation >= 0.999, "bodies still overlap: {separation}");
}

#[test]
fn falling_body_rests_on_ground() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut world = World::new();
    let mut sim = Simulation::new(
        gpu,
        4,
        PhysicsConfig {
            damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let ground = world.spawn((
        Transform::at([0.0, 0.0, 0.0]),
        Velocity::default(),
        Mass::static_body(),
        Restitution(0.0),
        SphereCollider::new(1.0),
    ));
    let ball = world.spawn((
        Transform::at([0.0, 3.0, 0.0]),
        Velocity::default(),
        Mass::new(1.0),
        Restitution(0.0),
        SphereCollider::new(0.5),
    ));
    sim.add(ground);
    sim.add(ball);
    for _ in 0..120 {
        sim.step(&world, DT);
        sim.sync_back(&mut world);
    }
    let ball_y = world.get::<Transform>(ball).unwrap().position[1];
    assert!(
        ball_y > 1.4 && ball_y < 1.6,
        "ball should rest on ground surface, got {ball_y}"
    );
}

#[test]
fn single_ball_rests_indefinitely() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut world = World::new();
    let mut sim = Simulation::new(
        gpu,
        4,
        PhysicsConfig {
            damping: 0.0,
            solve_iterations: 4,
            ..PhysicsConfig::default()
        },
    );
    let ground = world.spawn((
        Transform::at([0.0, 0.0, 0.0]),
        Velocity::default(),
        Mass::static_body(),
        Restitution(0.0),
        SphereCollider::new(1.0),
    ));
    let ball = world.spawn((
        Transform::at([0.0, 3.0, 0.0]),
        Velocity::default(),
        Mass::new(1.0),
        Restitution(0.0),
        SphereCollider::new(0.5),
    ));
    sim.add(ground);
    sim.add(ball);
    let mut lowest = f32::INFINITY;
    for _ in 0..600 {
        sim.step(&world, DT);
        sim.sync_back(&mut world);
        let y = world.get::<Transform>(ball).unwrap().position[1];
        lowest = lowest.min(y);
    }
    let final_y = world.get::<Transform>(ball).unwrap().position[1];
    assert!(
        final_y > 1.4 && final_y < 1.6,
        "ball should rest on ground surface, got {final_y}"
    );
    assert!(
        lowest > 1.4,
        "ball penetrated ground during resting, lowest was {lowest}"
    );
}

#[test]
fn stacked_bodies_do_not_collapse() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut world = World::new();
    let mut sim = Simulation::new(
        gpu,
        8,
        PhysicsConfig {
            damping: 0.0,
            solve_iterations: 8,
            ..PhysicsConfig::default()
        },
    );
    let ground = world.spawn((
        Transform::at([0.0, 0.0, 0.0]),
        Velocity::default(),
        Mass::static_body(),
        Restitution(0.0),
        SphereCollider::new(1.0),
    ));
    let lower = world.spawn((
        Transform::at([0.0, 1.5, 0.0]),
        Velocity::default(),
        Mass::new(1.0),
        Restitution(0.0),
        SphereCollider::new(0.5),
    ));
    let upper = world.spawn((
        Transform::at([0.0, 2.5, 0.0]),
        Velocity::default(),
        Mass::new(1.0),
        Restitution(0.0),
        SphereCollider::new(0.5),
    ));
    sim.add(ground);
    sim.add(lower);
    sim.add(upper);
    for _ in 0..120 {
        sim.step(&world, DT);
        sim.sync_back(&mut world);
    }
    let lower_y = world.get::<Transform>(lower).unwrap().position[1];
    let upper_y = world.get::<Transform>(upper).unwrap().position[1];
    assert!(
        lower_y > 1.4 && lower_y < 1.52,
        "lower body should rest on ground, got {lower_y}"
    );
    assert!(
        upper_y > 2.4 && upper_y < 2.6,
        "upper body should rest on lower body, got {upper_y}"
    );
}

#[test]
fn restitution_bounces_ball() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut world = World::new();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ground = world.spawn((
        Transform::at([0.0, 0.0, 0.0]),
        Velocity::default(),
        Mass::static_body(),
        Restitution(0.0),
        SphereCollider::new(1.0),
    ));
    let ball = world.spawn((
        Transform::at([0.0, 1.5, 0.0]),
        Velocity::at([0.0, -1.0, 0.0]),
        Mass::new(1.0),
        Restitution(0.8),
        SphereCollider::new(0.5),
    ));
    sim.add(ground);
    sim.add(ball);
    sim.step(&world, DT);
    sim.sync_back(&mut world);
    let velocity = world.get::<Velocity>(ball).unwrap().linear[1];
    assert!(
        velocity > 0.6,
        "ball should bounce upward, got velocity {velocity}"
    );
    let position = world.get::<Transform>(ball).unwrap().position[1];
    assert!(
        position > 1.4,
        "ball should stay above ground, got {position}"
    );
}

#[test]
fn sync_back_skips_despanned_body() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut world = World::new();
    let mut sim = Simulation::new(
        gpu,
        4,
        PhysicsConfig {
            damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let ball = world.spawn((
        Transform::at([0.0, 10.0, 0.0]),
        Velocity::default(),
        Mass::new(1.0),
        Restitution(0.0),
        SphereCollider::new(0.5),
    ));
    sim.add(ball);
    sim.step(&world, DT);
    world.despawn(ball);
    let count = sim.bodies().len();
    assert_eq!(count, 1);
    sim.sync_back(&mut world);
    assert_eq!(world.len(), 0);
}

#[test]
fn step_panics_when_body_lost_components() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut world = World::new();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = world.spawn((
        Transform::at([0.0, 0.0, 0.0]),
        Velocity::default(),
        Mass::new(1.0),
        Restitution(0.0),
        SphereCollider::new(0.5),
    ));
    sim.add(ball);
    world.remove::<Mass>(ball);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.step(&world, DT);
        }))
        .is_err()
    );
}

#[test]
fn step_panics_when_capacity_exceeded() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut world = World::new();
    let mut sim = Simulation::new(gpu, 2, static_config());
    let first = world.spawn((
        Transform::at([0.0, 0.0, 0.0]),
        Velocity::default(),
        Mass::new(1.0),
        Restitution(0.0),
        SphereCollider::new(0.5),
    ));
    let second = world.spawn((
        Transform::at([1.0, 0.0, 0.0]),
        Velocity::default(),
        Mass::new(1.0),
        Restitution(0.0),
        SphereCollider::new(0.5),
    ));
    sim.add(first);
    sim.add(second);
    let third = world.spawn((
        Transform::at([2.0, 0.0, 0.0]),
        Velocity::default(),
        Mass::new(1.0),
        Restitution(0.0),
        SphereCollider::new(0.5),
    ));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.add(third);
        }))
        .is_err()
    );
}
