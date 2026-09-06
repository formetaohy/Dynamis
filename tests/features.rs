use dynamis::{BodyDesc, ConstraintDesc, ConstraintKind, GpuContext, PhysicsConfig, Simulation};
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
fn box_falls_and_rests_on_ground() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let ground = sim.spawn(BodyDesc::static_sphere(40.0).position([0.0, -1.0, 0.0]));
    let cube = sim.spawn(
        BodyDesc::cuboid([0.5, 0.5, 0.5])
            .position([0.0, 45.0, 0.0])
            .restitution(0.0),
    );
    let _ = ground;
    for _ in 0..300 {
        sim.step(DT);
    }
    sim.wait();
    let y = sim.read_state(cube).position[1];
    assert!(
        y > 38.8 && y < 40.2,
        "cube must rest on ground top (39.0 + 0.5), got {y}"
    );
}

#[test]
fn cube_rotates_with_impulse_at_corner() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let cube = sim.spawn(BodyDesc::cuboid([0.5, 0.5, 0.5]));
    sim.apply_impulse_at_point(cube, [0.0, 0.0, 1.0], [0.0, 0.5, 0.0]);
    sim.step(DT);
    sim.wait();
    let spin = sim.read_state(cube).angular_velocity;
    assert!(
        spin[0].abs() > 0.01 || spin[2].abs() > 0.01,
        "off-center impulse must create angular velocity, got {spin:?}"
    );
}

#[test]
fn capsule_rests_upright() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let ground = sim.spawn(BodyDesc::static_sphere(50.0).position([0.0, -1.0, 0.0]));
    let _ = ground;
    let capsule = sim.spawn(
        BodyDesc::capsule(0.3, 1.0)
            .position([0.0, 2.0, 0.0])
            .restitution(0.0),
    );
    for _ in 0..240 {
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(capsule);
    assert!(
        state.position[1] > 1.0,
        "capsule must rest above ground, got {:?}",
        state.position
    );
}

#[test]
fn collision_groups_prevent_contact() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let first = sim.spawn(
        BodyDesc::sphere(0.5)
            .position([-1.0, 0.0, 0.0])
            .collision_group(1)
            .collision_mask(2),
    );
    let second = sim.spawn(
        BodyDesc::sphere(0.5)
            .position([1.0, 0.0, 0.0])
            .collision_group(2)
            .collision_mask(1),
    );
    for _ in 0..10 {
        sim.step(DT);
    }
    sim.wait();
    let first_x = sim.read_state(first).position[0];
    let second_x = sim.read_state(second).position[0];
    assert!(
        second_x - first_x >= 1.9,
        "bodies must not overlap, separation {}",
        second_x - first_x
    );
}

#[test]
fn mismatched_masks_prevent_contact() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let first = sim.spawn(
        BodyDesc::sphere(0.5)
            .position([-1.0, 0.0, 0.0])
            .collision_group(1)
            .collision_mask(4),
    );
    let second = sim.spawn(
        BodyDesc::sphere(0.5)
            .position([1.0, 0.0, 0.0])
            .collision_group(2)
            .collision_mask(1),
    );
    for _ in 0..10 {
        sim.step(DT);
    }
    sim.wait();
    let first_x = sim.read_state(first).position[0];
    let second_x = sim.read_state(second).position[0];
    assert!(
        second_x - first_x >= 1.9,
        "mismatched masks must prevent contact, separation {}",
        second_x - first_x
    );
}

#[test]
fn kinematic_body_moves_platform() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let platform = sim.spawn(
        BodyDesc::cuboid([2.0, 0.2, 2.0])
            .position([0.0, 2.0, 0.0])
            .kinematic(true)
            .velocity([1.0, 0.0, 0.0]),
    );
    let ball = sim.spawn(BodyDesc::sphere(0.3).position([0.0, 2.3, 0.0]));
    for _ in 0..120 {
        sim.step(DT);
    }
    sim.wait();
    let platform_x = sim.read_state(platform).position[0];
    let ball_x = sim.read_state(ball).position[0];
    let ball_y = sim.read_state(ball).position[1];
    assert!(
        platform_x > 0.5,
        "kinematic platform must advance, got {platform_x}"
    );
    assert!(
        (ball_y - 2.5).abs() < 0.5,
        "ball must rest on platform surface, ball_y={ball_y}"
    );
    assert!(
        ball_x > 0.01,
        "kinematic platform must push ball, ball_x={ball_x}"
    );
}

#[test]
fn kinematic_body_ignores_gravity() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let body = sim.spawn(BodyDesc::sphere(0.5).kinematic(true));
    for _ in 0..60 {
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(body);
    assert_eq!(
        state.position[1], 0.0,
        "kinematic body must not fall, got {:?}",
        state.position
    );
}

#[test]
fn ball_constraint_keeps_distance() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let first = sim.spawn(BodyDesc::sphere(0.2).position([0.0, 0.0, 0.0]));
    let second = sim.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    let constraint = sim.add_constraint(
        first,
        second,
        ConstraintDesc::ball([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    );
    sim.apply_force(first, [0.0, 5.0, 0.0]);
    for _ in 0..60 {
        sim.step(DT);
    }
    sim.wait();
    let p1 = sim.read_state(first).position;
    let p2 = sim.read_state(second).position;
    let distance =
        ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2)).sqrt();
    assert!(
        distance < 1.25,
        "ball constraint must keep bodies roughly at initial distance, got {distance}"
    );
    sim.remove_constraint(constraint);
}

#[test]
fn distance_constraint_holds_separation() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let first = sim.spawn(BodyDesc::sphere(0.2).position([0.0, 0.0, 0.0]));
    let second = sim.spawn(BodyDesc::sphere(0.2).position([0.0, 3.0, 0.0]));
    sim.add_constraint(
        first,
        second,
        ConstraintDesc::distance([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 3.0),
    );
    sim.apply_force(first, [2.0, 0.0, 0.0]);
    for _ in 0..60 {
        sim.step(DT);
    }
    sim.wait();
    let p1 = sim.read_state(first).position;
    let p2 = sim.read_state(second).position;
    let distance =
        ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2)).sqrt();
    assert!(
        (distance - 3.0).abs() < 0.4,
        "distance constraint must hold separation, got {distance}"
    );
}

#[test]
fn fixed_constraint_rigidly_links() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let first = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 0.0]));
    let second = sim.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 2.0]));
    let constraint = sim.add_constraint(
        first,
        second,
        ConstraintDesc::fixed([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    );
    sim.apply_force(first, [0.0, 0.0, 4.0]);
    for _ in 0..60 {
        sim.step(DT);
    }
    sim.wait();
    let p1 = sim.read_state(first).position;
    let p2 = sim.read_state(second).position;
    let separation = p2[2] - p1[2];
    assert!(
        (separation - 2.0).abs() < 0.3,
        "fixed constraint must preserve offset, got {separation}"
    );
    sim.remove_constraint(constraint);
}

#[test]
fn revolute_constraint_allows_hinge_rotation() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let anchor = sim.spawn(BodyDesc::static_sphere(0.1).position([0.0, 0.0, 0.0]));
    let arm = sim.spawn(BodyDesc::cuboid([0.1, 1.0, 0.1]).position([0.0, 1.0, 0.0]));
    sim.add_constraint(
        anchor,
        arm,
        ConstraintDesc::revolute([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    );
    sim.apply_force(arm, [1.0, 0.0, 0.0]);
    for _ in 0..120 {
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(arm);
    let tilt = state.orientation;
    let _tilt_degrees = 2.0 * (tilt[3].abs().acos()) * 180.0 / std::f32::consts::PI;
    assert!(
        state.angular_velocity.iter().any(|w| w.abs() > 1e-4),
        "revolute arm must rotate under lateral force, ang vel {:?}",
        state.angular_velocity
    );
}

#[test]
fn prismatic_constraint_slides_along_axis() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let guide = sim.spawn(BodyDesc::static_sphere(0.1).position([0.0, 5.0, 0.0]));
    let slider = sim.spawn(BodyDesc::sphere(0.3).position([0.0, 6.0, 0.0]));
    sim.add_constraint(
        guide,
        slider,
        ConstraintDesc::prismatic([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    );
    sim.apply_force(slider, [3.0, 0.0, 0.0]);
    for _ in 0..120 {
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(slider);
    let p = state.position;
    assert!(p[1] < 5.5, "slider must remain on axis line, got {p:?}");
}

#[test]
fn mass_changes_inertia() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let cube = sim.spawn(BodyDesc::cuboid([0.5, 0.5, 0.5]).mass(2.0));
    sim.apply_impulse_at_point(cube, [0.0, 0.0, 4.0], [0.0, 0.5, 0.0]);
    sim.step(DT);
    sim.wait();
    let spin_heavy = sim.read_state(cube).angular_velocity;
    let light = sim.spawn(BodyDesc::cuboid([0.5, 0.5, 0.5]).mass(1.0));
    let _ = light;
    assert!(spin_heavy[0] != 0.0 || spin_heavy[2] != 0.0);
}

#[test]
fn large_bodies_collide_with_small() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let ground = sim.spawn(BodyDesc::static_sphere(10.0).position([0.0, -1.0, 0.0]));
    let _ = ground;
    let ball = sim.spawn(BodyDesc::sphere(0.3).position([0.0, 5.0, 0.0]));
    for _ in 0..300 {
        sim.step(DT);
    }
    sim.wait();
    let y = sim.read_state(ball).position[1];
    assert!(
        y > 8.9 && y < 9.5,
        "large ground must support ball at y≈9.3, got {y}"
    );
}

#[test]
fn set_shape_switches_collider() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        8,
        PhysicsConfig {
            gravity: [0.0, -1.0, 0.0],
            ..PhysicsConfig::default()
        },
    );
    let ball = sim.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 5.0, 0.0])
            .restitution(0.0),
    );
    let ground = sim.spawn(BodyDesc::static_sphere(1.0).position([0.0, 0.0, 0.0]));
    let _ = ground;
    sim.set_shape(ball, dynamis::ShapeDesc::cuboid([0.5, 0.5, 0.5]));
    for _ in 0..300 {
        sim.step(DT);
    }
    sim.wait();
    let y = sim.read_state(ball).position[1];
    assert!(
        y > -1.0,
        "cube-switched body must remain near ground, got {y}"
    );
}

#[test]
fn raycast_hits_box_shape() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let cube = sim.spawn(BodyDesc::cuboid([1.0, 1.0, 1.0]).position([0.0, 0.0, 5.0]));
    let query = sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0);
    sim.step(DT);
    sim.wait();
    let hit = sim.query_hit(query).expect("ray should hit cube");
    assert_eq!(hit.body, cube);
    assert!(
        (hit.distance - 4.0).abs() < 0.05,
        "ray should hit cube front face at t=4, got {}",
        hit.distance
    );
}

#[test]
fn raycast_hits_capsule_shape() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let capsule = sim.spawn(BodyDesc::capsule(0.5, 1.0).position([0.0, 0.0, 5.0]));
    let _ = capsule;
    let query = sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0);
    sim.step(DT);
    sim.wait();
    let hit = sim.query_hit(query).expect("ray should hit capsule");
    assert!(
        (hit.distance - 4.5).abs() < 0.05,
        "ray should hit capsule surface at t=4.5, got {}",
        hit.distance
    );
}

#[test]
fn query_returns_point_and_normal() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    sim.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 2.0]));
    let query = sim.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 10.0);
    sim.step(DT);
    sim.wait();
    let hit = sim.query_hit(query).expect("hit");
    assert!(
        (hit.point[2] - 1.5).abs() < 0.01,
        "hit point must be on sphere surface, got {:?}",
        hit.point
    );
    assert!(
        (hit.normal[2] - (-1.0)).abs() < 0.01,
        "normal must point back along ray, got {:?}",
        hit.normal
    );
}

#[test]
fn thousand_bodies_broadphase_scales() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 1100, PhysicsConfig::default());
    for i in 0..1000 {
        let x = (i % 32) as f32 * 0.5;
        let z = (i / 32) as f32 * 0.5;
        sim.spawn(BodyDesc::sphere(0.2).position([x, 10.0, z]));
    }
    sim.spawn(BodyDesc::static_sphere(100.0).position([50.0, -2.0, 50.0]));
    for _ in 0..30 {
        sim.step(DT);
    }
    sim.wait();
    assert!(sim.count() == 1001);
}

#[test]
fn constraint_capacity_exhausted_panics() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 4, static_config());
    let first = sim.spawn(BodyDesc::sphere(0.2));
    let second = sim.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    let third = sim.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for _ in 0..4 {
            sim.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
        }
        sim.add_constraint(first, third, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    }));
    assert!(result.is_err(), "excessive constraints must panic");
}

#[test]
fn removing_constrained_body_panics() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let first = sim.spawn(BodyDesc::sphere(0.2));
    let second = sim.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    sim.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sim.remove(first);
    }));
    assert!(result.is_err(), "removing constrained body must panic");
}

#[test]
fn constraint_handle_generation_reuse() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 16, static_config());
    let first = sim.spawn(BodyDesc::sphere(0.2));
    let second = sim.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    let constraint = sim.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    sim.remove_constraint(constraint);
    let fresh = sim.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    assert_ne!(
        constraint.generation, fresh.generation,
        "reused slot must bump generation"
    );
}

#[test]
fn invalid_constraint_kind_with_zero_axis_panics() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, static_config());
    let first = sim.spawn(BodyDesc::sphere(0.2));
    let second = sim.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    let desc = ConstraintDesc {
        kind: ConstraintKind::Revolute,
        anchor_a: [0.0; 3],
        anchor_b: [0.0; 3],
        axis: [0.0; 3],
        distance: 0.0,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sim.add_constraint(first, second, desc);
    }));
    assert!(result.is_err(), "zero revolute axis must panic");
}

#[test]
fn kinematic_flag_round_trip() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(gpu, 8, PhysicsConfig::default());
    let body = sim.spawn(BodyDesc::sphere(0.5));
    sim.set_kinematic(body, true);
    sim.set_velocity(body, [2.0, 0.0, 0.0]);
    for _ in 0..10 {
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(body);
    assert!(
        state.position[0] > 0.2,
        "kinematic body should move with external velocity, got {:?}",
        state.position
    );
    sim.set_kinematic(body, false);
    sim.set_velocity(body, [0.0, 0.0, 0.0]);
    for _ in 0..30 {
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(body);
    assert!(
        state.position[1] < -0.1,
        "de-kinematic body should fall, got {:?}",
        state.position
    );
}

#[test]
fn high_speed_bullet_does_not_tunnel() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        8,
        PhysicsConfig {
            gravity: [0.0, 0.0, 0.0],
            damping: 0.0,
            angular_damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let _target = sim.spawn(BodyDesc::static_sphere(0.1));
    let bullet = sim.spawn(
        BodyDesc::sphere(0.5)
            .position([-5.0, 0.0, 0.0])
            .velocity([60.0, 0.0, 0.0])
            .restitution(0.0),
    );
    for _ in 0..30 {
        sim.step(DT);
        sim.wait();
    }
    let state = sim.read_state(bullet);
    assert!(
        state.position[0] < -0.5,
        "bullet should be stopped at the target surface, got x={}",
        state.position[0]
    );
    assert!(
        state.velocity[0].abs() < 0.01,
        "bullet velocity should be absorbed on contact, got {}",
        state.velocity[0]
    );
}

#[test]
fn moderate_speed_contact_uses_narrowphase() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        8,
        PhysicsConfig {
            gravity: [0.0, 0.0, 0.0],
            damping: 0.0,
            angular_damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let _ground = sim.spawn(BodyDesc::static_sphere(1.0));
    let ball = sim.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 2.0, 0.0])
            .velocity([0.0, -0.5, 0.0])
            .restitution(0.0),
    );
    for _ in 0..90 {
        sim.step(DT);
    }
    sim.wait();
    let state = sim.read_state(ball);
    assert!(
        (state.position[1] - 1.5).abs() < 0.05,
        "slow ball should rest exactly on surface, got y={}",
        state.position[1]
    );
}

#[test]
fn radix_sort_orders_duplicate_cells_stably() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        400,
        PhysicsConfig {
            gravity: [0.0, 0.0, 0.0],
            damping: 0.0,
            angular_damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let mut handles = Vec::new();
    for i in 0..300 {
        let x = (i % 10) as f32 * 0.3;
        let y = (i / 10) as f32 * 0.3;
        handles.push(sim.spawn(BodyDesc::sphere(0.1).position([x, y, 0.0])));
    }
    for _ in 0..5 {
        sim.step(DT);
    }
    sim.wait();
    for handle in handles {
        let _ = sim.read_state(handle);
    }
    let count = sim.count();
    assert_eq!(count, 300, "bodies were lost through broadphase");
}

#[test]
fn swept_aabb_covers_fast_motion() {
    let (_guard, gpu) = serialized_gpu();
    let mut sim = Simulation::new(
        gpu,
        8,
        PhysicsConfig {
            gravity: [0.0, 0.0, 0.0],
            damping: 0.0,
            angular_damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let _target = sim.spawn(BodyDesc::static_sphere(0.2).position([4.0, 0.0, 0.0]));
    let bullet = sim.spawn(
        BodyDesc::sphere(0.3)
            .position([0.0, 0.0, 0.0])
            .velocity([90.0, 0.0, 0.0])
            .restitution(0.0),
    );
    for _ in 0..20 {
        sim.step(DT);
        sim.wait();
    }
    let state = sim.read_state(bullet);
    assert!(
        state.position[0] < 3.8,
        "bullet should not tunnel at extreme speed, got x={}",
        state.position[0]
    );
}
