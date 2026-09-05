use dynamis_gpu::GpuContext;
use dynamis_physics::{BodyDesc, BodyHandle, PhysicsConfig, Simulation};

use std::sync::{Mutex, MutexGuard};

const GRAVITY: f32 = 9.81;
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

fn ground(sim: &mut Simulation, radius: f32) -> BodyHandle {
    sim.spawn(BodyDesc::static_sphere(radius))
}

#[test]
fn free_fall_matches_closed_form() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        4,
        PhysicsConfig {
            damping: 0.0,
            angular_damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let ball = sim.spawn(
        BodyDesc::sphere(0.1)
            .position([0.0, 10.0, 0.0])
            .mass(1.0)
            .restitution(0.0),
    );

    const STEPS: u32 = 30;
    for _ in 0..STEPS {
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(ball);
    let expected = 10.0 - 0.5 * GRAVITY * DT * DT * (STEPS as f32 * (STEPS as f32 + 1.0));
    assert!(
        (state.position[1] - expected).abs() < 1e-3,
        "expected {expected}, got {}",
        state.position[1]
    );
    let expected_velocity = -GRAVITY * DT * STEPS as f32;
    assert!((state.velocity[1] - expected_velocity).abs() < 1e-3);
}

#[test]
fn overlapping_bodies_separate() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let first = sim.spawn(BodyDesc::sphere(0.5).position([0.0, -0.4, 0.0]));
    let second = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 0.4, 0.0]));
    for _ in 0..16 {
        sim.step(DT);
    }
    sim.wait();
    let first_y = sim.read_state(first).position[1];
    let second_y = sim.read_state(second).position[1];
    let separation = second_y - first_y;
    assert!(separation >= 0.99, "bodies still overlap: {separation}");
}

#[test]
fn falling_body_rests_on_ground() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        4,
        PhysicsConfig {
            damping: 0.0,
            angular_damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let _ground = ground(&mut sim, 1.0);
    let ball = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 3.0, 0.0]));
    for _ in 0..120 {
        sim.step(DT);
    }
    sim.wait();
    let ball_y = sim.read_state(ball).position[1];
    assert!(
        ball_y > 1.4 && ball_y < 1.6,
        "ball should rest on ground surface, got {ball_y}"
    );
}

#[test]
fn single_ball_rests_indefinitely() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        4,
        PhysicsConfig {
            damping: 0.0,
            angular_damping: 0.0,
            solve_iterations: 4,
            ..PhysicsConfig::default()
        },
    );
    let _ground = ground(&mut sim, 1.0);
    let ball = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 3.0, 0.0]));
    let mut lowest = f32::INFINITY;
    for _ in 0..600 {
        sim.step(DT);
        sim.wait();
        let y = sim.read_state(ball).position[1];
        lowest = lowest.min(y);
    }
    let final_y = sim.read_state(ball).position[1];
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
    let mut sim = Simulation::new(
        gpu,
        8,
        PhysicsConfig {
            damping: 0.0,
            angular_damping: 0.0,
            solve_iterations: 8,
            ..PhysicsConfig::default()
        },
    );
    let _ground = ground(&mut sim, 1.0);
    let lower = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 1.5, 0.0]));
    let upper = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 2.5, 0.0]));
    for _ in 0..120 {
        sim.step(DT);
    }
    sim.wait();
    let lower_y = sim.read_state(lower).position[1];
    let upper_y = sim.read_state(upper).position[1];
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
    let mut sim = Simulation::new(gpu, 4, static_config());
    let _ground = ground(&mut sim, 1.0);
    let ball = sim.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 1.5, 0.0])
            .velocity([0.0, -2.0, 0.0])
            .restitution(0.8),
    );
    sim.step(DT);
    sim.wait();
    let state = sim.read_state(ball);
    assert!(
        state.velocity[1] > 1.2,
        "ball should bounce upward, got velocity {}",
        state.velocity[1]
    );
    assert!(
        state.position[1] > 1.4,
        "ball should stay above ground, got {}",
        state.position[1]
    );
}

#[test]
fn slow_contact_does_not_bounce() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let _ground = ground(&mut sim, 1.0);
    let ball = sim.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 1.5, 0.0])
            .velocity([0.0, -0.5, 0.0])
            .restitution(0.9),
    );
    sim.step(DT);
    sim.wait();
    let state = sim.read_state(ball);
    assert!(
        state.velocity[1] > -0.05 && state.velocity[1] < 0.01,
        "restitution threshold should suppress micro-bounce, got velocity {}",
        state.velocity[1]
    );
}

#[test]
fn removed_body_leaves_others_intact() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let first = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 0.0]));
    let second = sim.spawn(BodyDesc::sphere(0.5).position([2.0, 0.0, 0.0]));
    let third = sim.spawn(BodyDesc::sphere(0.5).position([4.0, 0.0, 0.0]));
    sim.step(DT);
    sim.remove(second);
    sim.step(DT);
    sim.wait();
    assert_eq!(sim.count(), 2);
    let first_x = sim.read_state(first).position[0];
    let third_x = sim.read_state(third).position[0];
    assert_eq!(first_x, 0.0, "first body was displaced by removal");
    assert_eq!(third_x, 4.0, "third body was displaced by removal");
}

#[test]
fn removed_body_read_panics() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = sim.spawn(BodyDesc::sphere(0.5));
    sim.step(DT);
    sim.remove(ball);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.read_state(ball);
        }))
        .is_err()
    );
}

#[test]
fn stale_handle_panics() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let old = sim.spawn(BodyDesc::sphere(0.5));
    sim.remove(old);
    let fresh = sim.spawn(BodyDesc::sphere(0.5));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.remove(old);
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.read_state(old);
        }))
        .is_err()
    );
    sim.step(DT);
    assert_eq!(sim.count(), 1);
    assert_eq!(sim.read_state(fresh).position, [0.0, 0.0, 0.0]);
}

#[test]
fn set_velocity_overrides_simulation() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        4,
        PhysicsConfig {
            damping: 0.0,
            angular_damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let ball = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 5.0, 0.0]));
    sim.step(DT);
    sim.set_velocity(ball, [0.0, 3.0, 0.0]);
    sim.step(DT);
    sim.wait();
    let state = sim.read_state(ball);
    let expected_velocity = 3.0 - GRAVITY * DT;
    assert!(
        (state.velocity[1] - expected_velocity).abs() < 1e-4,
        "patched velocity was not applied, got {} (expected {expected_velocity})",
        state.velocity[1]
    );
    let expected_position = 5.0 + 3.0 * DT - 2.0 * GRAVITY * DT * DT;
    assert!(
        (state.position[1] - expected_position).abs() < 1e-3,
        "position should follow the patched velocity, got {} (expected {expected_position})",
        state.position[1]
    );
}

#[test]
fn set_position_teleports_body() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        4,
        PhysicsConfig {
            damping: 0.0,
            angular_damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let ball = sim.spawn(BodyDesc::sphere(0.5));
    sim.step(DT);
    sim.set_position(ball, [7.0, 1.0, 2.0]);
    sim.step(DT);
    sim.wait();
    let state = sim.read_state(ball);
    assert!((state.position[0] - 7.0).abs() < 1e-4);
    assert!((state.position[1] - (1.0 - 2.0 * GRAVITY * DT * DT)).abs() < 1e-4);
    assert!((state.position[2] - 2.0).abs() < 1e-4);
}

#[test]
fn set_mass_freezes_body() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        4,
        PhysicsConfig {
            damping: 0.0,
            angular_damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let ball = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 3.0, 0.0]));
    sim.step(DT);
    sim.set_mass(ball, 0.0);
    sim.step(DT);
    sim.wait();
    let state = sim.read_state(ball);
    assert_eq!(state.inverse_mass, 0.0);
    assert_eq!(state.velocity, [0.0, 0.0, 0.0]);
    assert_eq!(state.angular_velocity, [0.0, 0.0, 0.0]);
    let frozen_y = sim.read_state(ball).position[1];
    assert_eq!(state.position[1], frozen_y);
}

#[test]
fn despawned_mid_step_command_batch() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let first = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 0.0]));
    let second = sim.spawn(BodyDesc::sphere(0.5).position([2.0, 0.0, 0.0]));
    sim.remove(first);
    let third = sim.spawn(BodyDesc::sphere(0.5).position([4.0, 0.0, 0.0]));
    sim.step(DT);
    sim.wait();
    assert_eq!(sim.count(), 2);
    assert_eq!(sim.read_state(second).position, [2.0, 0.0, 0.0]);
    assert_eq!(sim.read_state(third).position, [4.0, 0.0, 0.0]);
}

#[test]
fn step_panics_on_non_positive_dt() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.step(0.0);
        }))
        .is_err()
    );
}

#[test]
fn step_panics_when_capacity_exceeded() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 2, static_config());
    let _first = sim.spawn(BodyDesc::sphere(0.5));
    let _second = sim.spawn(BodyDesc::sphere(0.5));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.spawn(BodyDesc::sphere(0.5));
        }))
        .is_err()
    );
}

#[test]
fn freshly_spawned_state_available_immediately() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = sim.spawn(
        BodyDesc::sphere(0.5)
            .position([1.0, 2.0, 3.0])
            .velocity([4.0, 5.0, 6.0]),
    );
    let state = sim.read_state(ball);
    assert_eq!(state.position, [1.0, 2.0, 3.0]);
    assert_eq!(state.velocity, [4.0, 5.0, 6.0]);
}

#[test]
fn readback_returns_latest_step() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        4,
        PhysicsConfig {
            damping: 0.0,
            angular_damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let ball = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 10.0, 0.0]));
    const STEPS: u32 = 10;
    for _ in 0..STEPS {
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(ball);
    let expected = 10.0 - 0.5 * GRAVITY * DT * DT * (STEPS as f32 * (STEPS as f32 + 1.0));
    assert!(
        (state.position[1] - expected).abs() < 1e-3,
        "readback should return the most recent step, got {}",
        state.position[1]
    );
}

#[test]
fn apply_force_accelerates_body() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = sim.spawn(BodyDesc::sphere(0.5).mass(2.0));
    const STEPS: u32 = 10;
    for _ in 0..STEPS {
        sim.apply_force(ball, [0.0, 4.0, 0.0]);
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(ball);
    let expected = 4.0 * 0.5 * DT * STEPS as f32;
    assert!(
        (state.velocity[1] - expected).abs() < 1e-3,
        "force should accelerate the body, got {} (expected {expected})",
        state.velocity[1]
    );
}

#[test]
fn force_is_consumed_within_one_step() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = sim.spawn(BodyDesc::sphere(0.5).mass(2.0));
    sim.apply_force(ball, [0.0, 4.0, 0.0]);
    sim.step(DT);
    sim.wait();
    let after_force = sim.read_state(ball).velocity[1];
    for _ in 0..5 {
        sim.step(DT);
    }
    sim.wait();
    let after_idle = sim.read_state(ball).velocity[1];
    assert_eq!(
        after_force, after_idle,
        "force should only affect the step it was applied"
    );
}

#[test]
fn apply_torque_spins_body() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = sim.spawn(BodyDesc::sphere(0.5));
    const STEPS: u32 = 5;
    for _ in 0..STEPS {
        sim.apply_torque(ball, [0.0, 0.0, 5.0]);
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(ball);
    let inverse_inertia = 2.5 / (0.5 * 0.5);
    let expected = inverse_inertia * 5.0 * DT * STEPS as f32;
    assert!(
        (state.angular_velocity[2] - expected).abs() < 1e-3,
        "torque should spin the body, got {} (expected {expected})",
        state.angular_velocity[2]
    );
}

#[test]
fn apply_impulse_changes_velocity() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = sim.spawn(BodyDesc::sphere(0.5).mass(2.0));
    sim.apply_impulse(ball, [3.0, 0.0, 0.0]);
    sim.step(DT);
    sim.wait();
    let state = sim.read_state(ball);
    assert!(
        (state.velocity[0] - 1.5).abs() < 1e-4,
        "impulse should change velocity, got {}",
        state.velocity[0]
    );
}

#[test]
fn apply_impulse_at_point_spins_body() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = sim.spawn(BodyDesc::sphere(0.5));
    sim.apply_impulse_at_point(ball, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
    sim.step(DT);
    sim.wait();
    let state = sim.read_state(ball);
    let inverse_inertia = 2.5 / (0.5 * 0.5);
    assert!(
        (state.angular_velocity[2] - inverse_inertia).abs() < 1e-3,
        "offset impulse should spin the body, got {} (expected {inverse_inertia})",
        state.angular_velocity[2]
    );
}

#[test]
fn initial_angular_velocity_rotates_body() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = sim.spawn(BodyDesc::sphere(0.5).angular_velocity([0.0, 0.0, 3.0]));
    const STEPS: u32 = 30;
    for _ in 0..STEPS {
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(ball);
    assert!(
        state.orientation[2] > 0.5,
        "body should have rotated, got orientation {:?}",
        state.orientation
    );
    let norm = state
        .orientation
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "orientation must stay unit length, got norm {norm}"
    );
}

#[test]
fn friction_converts_slip_into_roll() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        4,
        PhysicsConfig {
            gravity: [0.0, -GRAVITY, 0.0],
            damping: 0.0,
            angular_damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let _ground = ground(&mut sim, 100.0);
    let ball = sim.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 100.5, 0.0])
            .velocity([2.0, 0.0, 0.0])
            .friction(0.0),
    );
    for _ in 0..30 {
        sim.step(DT);
    }
    sim.wait();
    let slipping = sim.read_state(ball);
    assert!(
        slipping.velocity[0] > 1.9,
        "frictionless body should keep sliding without slowing, got {}",
        slipping.velocity[0]
    );
    assert!(
        slipping.angular_velocity[2].abs() < 0.01,
        "frictionless body should not spin, got {}",
        slipping.angular_velocity[2]
    );

    sim.set_friction(ball, 0.8);
    for _ in 0..180 {
        sim.step(DT);
    }
    sim.wait();
    let rolling = sim.read_state(ball);
    assert!(
        rolling.angular_velocity[2] < -1.5,
        "friction should spin the rolling ball, got {}",
        rolling.angular_velocity[2]
    );
    assert!(
        (rolling.velocity[0] + rolling.angular_velocity[2] * 0.5).abs() < 0.1,
        "ball should roll without slipping, got v={} omega={}",
        rolling.velocity[0],
        rolling.angular_velocity[2]
    );
}

#[test]
fn rotational_patches_apply() {
    const HALF_SQRT_TWO: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = sim.spawn(BodyDesc::sphere(0.5));
    sim.step(DT);
    sim.set_angular_velocity(ball, [0.0, 1.0, 0.0]);
    sim.set_orientation(ball, [0.0, HALF_SQRT_TWO, 0.0, HALF_SQRT_TWO]);
    sim.step(DT);
    sim.wait();
    let state = sim.read_state(ball);
    assert!(
        (state.angular_velocity[1] - 1.0).abs() < 1e-4,
        "angular velocity patch was not applied, got {:?}",
        state.angular_velocity
    );
    let half_angle = 0.5f32 / 60.0;
    let y_expected = (HALF_SQRT_TWO * (1.0 + half_angle)) / (1.0 + half_angle * half_angle).sqrt();
    assert!(
        (state.orientation[1] - y_expected).abs() < 1e-3,
        "orientation patch was not applied or was lost, got {:?}",
        state.orientation
    );
}

#[test]
fn set_orientation_panics_on_non_unit_quaternion() {
    let (_gpu_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let ball = sim.spawn(BodyDesc::sphere(0.5));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.set_orientation(ball, [1.0, 1.0, 0.0, 0.0]);
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            BodyDesc::sphere(0.5).orientation([1.0, 1.0, 0.0, 0.0]);
        }))
        .is_err()
    );
}
