use dynamis::{BodyDesc, ConstraintDesc, GpuContext, PhysicsConfig, Simulation};
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
        sleep_velocity: 0.0,
        sleep_angular_velocity: 0.0,
        ..PhysicsConfig::default()
    }
}

fn hinge_angle(orientation: [f32; 4]) -> f32 {
    let quat = [orientation[3], orientation[0], orientation[1], orientation[2]];
    let w = quat[0];
    let y = quat[2];
    (2.0 * (y * w).atan2(w * w - y * y)).abs()
}

#[test]
fn revolute_limit_stops_rotation() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let anchor = sim.spawn(BodyDesc::static_sphere(0.1).position([0.0, 0.0, 0.0]));
    let arm = sim.spawn(BodyDesc::cuboid([0.1, 1.0, 0.1]).position([0.0, 1.0, 0.0]));
    sim.add_constraint(
        anchor,
        arm,
        ConstraintDesc::revolute([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0])
            .limit(-0.2, 0.2),
    );
    for _ in 0..120 {
        sim.apply_force(arm, [1.0, 0.0, 0.0]);
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(arm);
    let angle = hinge_angle(state.orientation);
    assert!(
        angle < 0.25,
        "revolute limit must cap the swing, got {angle} rad"
    );
}

#[test]
fn prismatic_motor_drives_slider() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let anchor = sim.spawn(BodyDesc::static_sphere(0.1).position([0.0, 0.0, 0.0]));
    let slider = sim.spawn(BodyDesc::cuboid([0.1, 0.1, 0.1]).position([1.0, 0.0, 0.0]));
    sim.add_constraint(
        anchor,
        slider,
        ConstraintDesc::prismatic([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0])
            .motor(2.0),
    );
    for _ in 0..60 {
        sim.step(DT);
    }
    sim.wait();
    let x = sim.read_state(slider).position[0];
    assert!(
        x > 2.5,
        "motor must drive the slider along its axis, got x={x}"
    );
}

#[test]
fn prismatic_limit_stops_slider() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let anchor = sim.spawn(BodyDesc::static_sphere(0.1).position([0.0, 0.0, 0.0]));
    let slider = sim.spawn(BodyDesc::cuboid([0.1, 0.1, 0.1]).position([1.0, 0.0, 0.0]));
    sim.add_constraint(
        anchor,
        slider,
        ConstraintDesc::prismatic([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0])
            .limit(0.0, 2.0)
            .motor(2.0),
    );
    for _ in 0..90 {
        sim.step(DT);
    }
    sim.wait();
    let x = sim.read_state(slider).position[0];
    assert!(
        (x - 1.5).abs() < 0.05,
        "prismatic limit must cap the travel at 1.5, got x={x}"
    );
}

#[test]
fn distance_spring_sags_under_gravity() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        8,
        PhysicsConfig {
            sleep_velocity: 0.0,
            sleep_angular_velocity: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let anchor = sim.spawn(BodyDesc::static_sphere(0.1).position([0.0, 0.0, 0.0]));
    let ball = sim.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]));
    sim.add_constraint(
        anchor,
        ball,
        ConstraintDesc::distance([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0).spring(1.0, 0.5),
    );
    for _ in 0..240 {
        sim.step(DT);
    }
    sim.wait();
    let y = sim.read_state(ball).position[1];
    assert!(
        y > 0.3 && y < 1.0,
        "spring must sag below the rest length, got y={y}"
    );
}

#[test]
fn joint_disable_collisions_holds_overlap() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let first = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 0.0]));
    let second = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 0.6, 0.0]));
    sim.add_constraint(
        first,
        second,
        ConstraintDesc::ball([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    );
    for _ in 0..5 {
        sim.step(DT);
    }
    sim.wait();
    let y = sim.read_state(second).position[1];
    assert!(
        y < 0.75,
        "disabled collisions must not push joined bodies apart, got y={y}"
    );
}

#[test]
fn joint_collisions_push_apart_when_enabled() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let first = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 0.0]));
    let second = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 0.6, 0.0]));
    sim.add_constraint(
        first,
        second,
        ConstraintDesc::ball([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]).disable_collisions(false),
    );
    for _ in 0..30 {
        sim.step(DT);
    }
    sim.wait();
    let y = sim.read_state(second).position[1];
    assert!(
        y > 0.75,
        "enabled collisions must push joined bodies apart, got y={y}"
    );
}
