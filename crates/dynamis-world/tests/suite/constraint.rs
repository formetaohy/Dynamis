use super::common::{DT, converged, distance, new_world, settle, settle_until, static_config};
use dynamis_model::{BodyDesc, ConstraintDesc, ConstraintMotor, DofDesc, PhysicsConfig};
use dynamis_world::World;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn hinge_angle(orientation: [f32; 4]) -> f32 {
    let w = orientation[3];
    let z = orientation[2];
    (2.0 * (z * w).atan2(w * w - z * z)).abs()
}

#[test]
fn ball_constraint_keeps_bodies_linked() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    world.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    for _ in 0..60 {
        world.apply_force(first, [0.0, 5.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let span = distance(
        world.read_state(first).position,
        world.read_state(second).position,
    );
    assert!(
        span < 1.25,
        "ball constraint must keep bodies near their initial distance, got {span}"
    );
}

#[test]
fn distance_constraint_holds_span() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([0.0, 3.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 3.0),
    );
    for _ in 0..60 {
        world.apply_force(first, [2.0, 0.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let span = distance(
        world.read_state(first).position,
        world.read_state(second).position,
    );
    assert!(
        (span - 3.0).abs() < 0.4,
        "distance constraint must hold the span, got {span}"
    );
}

#[test]
fn fixed_constraint_preserves_offset() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.5));
    let second = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 2.0]));
    world.add_constraint(first, second, ConstraintDesc::fixed([0.0; 3], [0.0; 3]));
    world.apply_force(first, [0.0, 0.0, 4.0]);
    settle(&mut world, 60);
    let separation = world.read_state(second).position[2] - world.read_state(first).position[2];
    assert!(
        (separation - 2.0).abs() < 0.3,
        "fixed constraint must preserve the offset, got {separation}"
    );
}

fn pendulum_world(
    limit: Option<(f32, f32)>,
    motor: Option<f32>,
) -> (World, dynamis_model::BodyHandle) {
    let mut world = new_world(PhysicsConfig {
        sleep_velocity: 0.0,
        sleep_angular_velocity: 0.0,
        ..static_config()
    });
    let anchor = world.spawn(BodyDesc::static_sphere(0.1));
    let arm = world.spawn(BodyDesc::cuboid([0.1, 1.0, 0.1]).position([0.0, 1.0, 0.0]));
    let mut desc = ConstraintDesc::revolute([0.0; 3], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]);
    if let Some((min, max)) = limit {
        desc = desc.limit(min, max);
    }
    if let Some(speed) = motor {
        desc = desc.motor(speed);
    }
    world.add_constraint(anchor, arm, desc);
    (world, arm)
}

#[test]
fn revolute_rotates_freely_without_limit() {
    let (mut world, arm) = pendulum_world(None, None);
    world.apply_force(arm, [1.0, 0.0, 0.0]);
    let mut max_angle = 0.0f32;
    settle_until(&mut world, 120, |world| {
        max_angle = max_angle.max(hinge_angle(world.read_state(arm).orientation));
        max_angle > 0.01
    });
    assert!(
        max_angle > 0.01,
        "unlimited revolute must swing under a kick, max {max_angle} rad"
    );
}

#[test]
fn revolute_limit_caps_swing() {
    let (mut world, arm) = pendulum_world(Some((-0.2, 0.2)), None);
    for _ in 0..120 {
        world.apply_force(arm, [1.0, 0.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let angle = hinge_angle(world.read_state(arm).orientation);
    assert!(
        angle < 0.25,
        "revolute limit must cap the swing, got {angle} rad"
    );
}

#[test]
fn revolute_motor_drives_arm() {
    let (mut world, arm) = pendulum_world(None, Some(2.0));
    settle_until(&mut world, 90, |world| {
        hinge_angle(world.read_state(arm).orientation) > 1.0
    });
    let angle = hinge_angle(world.read_state(arm).orientation);
    assert!(
        angle > 1.0,
        "revolute motor must drive the arm, got {angle} rad"
    );
}

#[test]
fn prismatic_locks_perpendicular_motion() {
    let mut world = new_world(static_config());
    let guide = world.spawn(BodyDesc::static_sphere(0.1).position([0.0, 5.0, 0.0]));
    let slider = world.spawn(BodyDesc::sphere(0.3).position([0.0, 6.0, 0.0]));
    world.add_constraint(
        guide,
        slider,
        ConstraintDesc::prismatic([0.0; 3], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    );
    world.apply_force(slider, [3.0, 0.0, 0.0]);
    settle(&mut world, 120);
    let position = world.read_state(slider).position;
    assert!(
        (position[1] - 6.0).abs() < 0.15,
        "prismatic must hold the axis offset, got {:?}",
        position
    );
    assert!(
        position[0].abs() < 0.2,
        "prismatic must resist cross-axis force, got {:?}",
        position
    );
}

#[test]
fn prismatic_motor_drives_and_limit_caps_travel() {
    let mut world = new_world(static_config());
    let anchor = world.spawn(BodyDesc::static_sphere(0.1));
    let slider = world.spawn(BodyDesc::cuboid([0.1, 0.1, 0.1]).position([1.0, 0.0, 0.0]));
    world.add_constraint(
        anchor,
        slider,
        ConstraintDesc::prismatic([0.0; 3], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0])
            .limit(0.0, 2.0)
            .motor(2.0),
    );
    for _ in 0..90 {
        world.step(DT);
    }
    world.wait();
    let x = world.read_state(slider).position[0];
    assert!(
        (x - 1.5).abs() < 0.05,
        "prismatic limit must cap the travel at 1.5, got x={x}"
    );
}

#[test]
fn prismatic_motor_alone_drives_beyond_start() {
    let mut world = new_world(static_config());
    let anchor = world.spawn(BodyDesc::static_sphere(0.1));
    let slider = world.spawn(BodyDesc::cuboid([0.1, 0.1, 0.1]).position([1.0, 0.0, 0.0]));
    world.add_constraint(
        anchor,
        slider,
        ConstraintDesc::prismatic([0.0; 3], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]).motor(2.0),
    );
    for _ in 0..60 {
        world.step(DT);
    }
    world.wait();
    let x = world.read_state(slider).position[0];
    assert!(
        x > 2.5,
        "motor must drive the slider along its axis, got x={x}"
    );
}

#[test]
fn distance_spring_sags_under_gravity() {
    let mut world = new_world(PhysicsConfig {
        sleep_velocity: 0.0,
        sleep_angular_velocity: 0.0,
        ..PhysicsConfig::default()
    });
    let anchor = world.spawn(BodyDesc::static_sphere(0.1));
    let ball = world.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]));
    world.add_constraint(
        anchor,
        ball,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0).spring(1.0, 0.5),
    );
    let mut previous = f32::INFINITY;
    settle_until(&mut world, 240, |world| {
        converged(&mut previous, world.read_state(ball).position[1])
    });
    let y = world.read_state(ball).position[1];
    assert!(
        y > 0.3 && y < 1.0,
        "spring must sag below the rest length, got y={y}"
    );
}

#[test]
fn joined_bodies_collision_policy_controls_overlap() {
    let mut merged = new_world(static_config());
    let first = merged.spawn(BodyDesc::sphere(0.5));
    let second = merged.spawn(BodyDesc::sphere(0.5).position([0.0, 0.6, 0.0]));
    merged.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    settle(&mut merged, 5);
    let y = merged.read_state(second).position[1];
    assert!(
        y < 0.75,
        "disabled collisions must not push joined bodies apart, got y={y}"
    );

    let mut separated = new_world(static_config());
    let first = separated.spawn(BodyDesc::sphere(0.5));
    let second = separated.spawn(BodyDesc::sphere(0.5).position([0.0, 0.6, 0.0]));
    separated.add_constraint(
        first,
        second,
        ConstraintDesc::ball([0.0; 3], [0.0; 3]).disable_collisions(false),
    );
    settle(&mut separated, 30);
    let y = separated.read_state(second).position[1];
    assert!(
        y > 0.75,
        "enabled collisions must push joined bodies apart, got y={y}"
    );
}

#[test]
fn constraint_handle_reuse_bumps_generation() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    let constraint = world.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    world.remove_constraint(constraint);
    let fresh = world.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    assert_ne!(constraint.generation, fresh.generation);
    assert_eq!(world.constraints(), &[fresh]);
    world.remove_constraint(fresh);
    world.remove(first);
    assert_eq!(world.count(), 1);
}

#[test]
fn constraint_misuse_panics() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    let joint = world.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.remove(first))).is_err(),
        "removing a constrained body must panic"
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.add_constraint(first, first, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
        }))
        .is_err(),
        "a constraint must join distinct bodies"
    );
    world.remove_constraint(joint);
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.remove_constraint(joint))).is_err(),
        "a removed constraint handle must stay stale"
    );
}

#[test]
fn removing_constraint_allows_body_removal() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    let constraint = world.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    world.remove_constraint(constraint);
    world.remove(first);
    assert_eq!(world.count(), 1);
    assert!(world.constraints().is_empty());
}

fn twist_angle(orientation: [f32; 4]) -> f32 {
    let q = orientation;
    let y = q[1];
    let w = q[3];
    (2.0 * (y * w).atan2(w * w - y * y)).abs()
}

#[test]
fn ball_twist_limit_caps_relative_rotation() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::ball([0.0; 3], [0.0; 3])
            .axis([0.0, 0.0, 1.0])
            .limit(-0.3, 0.3),
    );
    let mut previous = f32::INFINITY;
    for frame in 1usize..=120 {
        world.apply_torque(second, [0.0, 0.0, 12.0]);
        world.step(DT);
        if frame.is_multiple_of(4) {
            world.wait();
            let angle = twist_angle(world.read_state(second).orientation);
            if converged(&mut previous, angle) {
                break;
            }
        }
    }
    world.wait();
    let state = world.read_state(second);
    let angle = twist_angle(state.orientation);
    assert!(
        angle < 0.5,
        "twist limit must cap relative rotation, got {angle} rad"
    );
}

#[test]
fn ball_swing_limit_caps_conical_sway() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::ball([0.0; 3], [0.0; 3]).swing(0.2, 0.2),
    );
    for _ in 0..120 {
        world.apply_torque(second, [0.0, 20.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let state = world.read_state(second);
    let q = state.orientation;
    let axis = 2.0 * (q[0] * q[0] + q[2] * q[2]).sqrt();
    let tilt = 2.0 * axis.clamp(0.0, 1.0).asin();
    println!("swing q={q:?} tilt={tilt}");
    assert!(
        tilt < 0.45,
        "swing limit must cap the sway angle, got {tilt} rad"
    );
}

#[test]
fn gear_constraint_links_angular_velocities() {
    let mut world = new_world(static_config());
    let _anchor = world.spawn(BodyDesc::static_sphere(0.1));
    let first = world.spawn(BodyDesc::sphere(0.3).position([0.0, 0.0, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.3).position([1.0, 0.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::gear([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 2.0),
    );
    world.set_angular_velocity(first, [0.0, 0.0, 10.0]);
    for _ in 0..60 {
        world.step(DT);
    }
    world.wait();
    let a = world.read_state(first).angular_velocity[2];
    let b = world.read_state(second).angular_velocity[2];
    assert!(
        (b - 2.0 * a).abs() < 1.0,
        "gear must enforce omega_b = 2 * omega_a, got a={a} b={b}"
    );
}

#[test]
fn pulley_constraint_holds_rope_length() {
    let mut world = new_world(static_config());
    let _anchor = world.spawn(BodyDesc::static_sphere(0.1));
    let first = world.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.2).position([2.0, 1.0, 0.0]));
    let fixed_a = [0.0f32, 3.0, 0.0];
    let fixed_b = [2.0f32, 3.0, 0.0];
    let length = 2.0 + 2.0;
    world.add_constraint(
        first,
        second,
        ConstraintDesc::pulley([0.0; 3], [0.0; 3], fixed_a, fixed_b, length),
    );
    for _ in 0..90 {
        world.apply_force(first, [0.0, 6.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let pa = world.read_state(first).position;
    let pb = world.read_state(second).position;
    let rope = distance(pa, fixed_a) + distance(pb, fixed_b);
    assert!(
        (rope - length).abs() < 0.3,
        "pulley must hold the rope length, got {rope} vs {length}"
    );
}

#[test]
fn break_threshold_removes_constraint_under_load() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.3));
    let second = world.spawn(BodyDesc::sphere(0.3).position([1.0, 0.0, 0.0]));
    let handle = world.add_constraint(
        first,
        second,
        ConstraintDesc::ball([0.0; 3], [0.3, 0.0, 0.0]).break_threshold(0.4, 0.0),
    );
    for frame in 1usize..=90 {
        world.apply_force(second, [0.0, 40.0, 0.0]);
        world.step(DT);
        if frame.is_multiple_of(4) {
            world.wait();
            let escaped = world.read_state(second).position[1] > 1.1;
            if !world.constraints().contains(&handle) && escaped {
                break;
            }
        }
    }
    world.wait();
    assert!(
        !world.constraints().contains(&handle),
        "overloaded ball constraint must break and be removed"
    );
    let state = world.read_state(second);
    assert!(
        state.position[1] > 1.1,
        "broken body must be free to move, got {:?}",
        state.position
    );
}

#[test]
fn motor_force_cap_limits_driving_torque() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.3));
    let second = world.spawn(BodyDesc::sphere(0.3).position([1.0, 0.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::revolute([0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
            .motor(20.0)
            .motor_force(0.5),
    );
    let mut previous = f32::INFINITY;
    settle_until(&mut world, 120, |world| {
        converged(&mut previous, world.read_state(second).angular_velocity[2])
    });
    let spin = world.read_state(second).angular_velocity[2].abs();
    assert!(
        spin < 6.0,
        "a tiny motor force must cap the driven spin, got {spin}"
    );
}

#[test]
fn cone_constraint_caps_swing_angle() {
    let mut world = new_world(static_config());
    let anchor = world.spawn(BodyDesc::sphere(0.1).mass(0.0));
    let tip = world.spawn(BodyDesc::sphere(0.1).position([-2.0, 0.0, 0.0]));
    world.add_constraint(
        anchor,
        tip,
        ConstraintDesc::cone([0.0; 3], [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 0.35)
            .axis_b([1.0, 0.0, 0.0]),
    );
    world.set_velocity(tip, [0.0, 1.2, 0.0]);
    for _ in 0..240 {
        world.step(DT);
    }
    world.wait();
    let state = world.read_state(tip);
    let tilt = (state.position[1].atan2(-state.position[0])).abs();
    assert!(tilt < 0.45, "cone must cap the swing tilt, got {tilt}");
}

#[test]
fn six_dof_locked_acts_as_fixed_constraint() {
    let mut world = new_world(static_config());
    let base = world.spawn(BodyDesc::sphere(0.5));
    let link = world.spawn(BodyDesc::sphere(0.5).position([1.5, 0.0, 0.0]));
    let desc = ConstraintDesc::six_dof([0.0; 3], [1.5, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        .dofs([DofDesc::locked(); 6]);
    world.add_constraint(base, link, desc);
    for _ in 0..120 {
        world.step(DT);
    }
    world.wait();
    let base_state = world.read_state(base);
    let link_state = world.read_state(link);
    let offset = [
        link_state.position[0] - base_state.position[0] + 1.5,
        link_state.position[1] - base_state.position[1],
        link_state.position[2] - base_state.position[2],
    ];
    assert!(
        distance(offset, [0.0, 0.0, 0.0]) < 0.05,
        "locked six dof must keep anchors coincident"
    );
    world.set_velocity(base, [0.4, 0.0, 0.0]);
    for _ in 0..120 {
        world.step(DT);
    }
    world.wait();
    let carried = distance(
        world.read_state(link).position,
        [
            world.read_state(base).position[0] - 1.5,
            world.read_state(base).position[1],
            world.read_state(base).position[2],
        ],
    );
    assert!(
        carried < 0.08,
        "locked six dof must carry the link with the base"
    );
}

#[test]
fn six_dof_linear_limit_caps_separation() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    let dofs = [
        DofDesc::free(),
        DofDesc::limited(-0.55, -0.45),
        DofDesc::free(),
        DofDesc::locked(),
        DofDesc::locked(),
        DofDesc::locked(),
    ];
    let desc = ConstraintDesc::six_dof([0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        .dofs(dofs);
    world.add_constraint(first, second, desc);
    let mut previous = f32::INFINITY;
    settle_until(&mut world, 180, |world| {
        let separation = world.read_state(second).position[0] - world.read_state(first).position[0];
        converged(&mut previous, separation)
    });
    let separation = world.read_state(second).position[0] - world.read_state(first).position[0];
    assert!(
        separation.abs() > 0.38 && separation.abs() < 0.62,
        "linear dof limit must pull the separation to its bound, got {separation}"
    );
}

#[test]
fn six_dof_servo_spins_to_target() {
    let mut world = new_world(static_config());
    let anchor = world.spawn(BodyDesc::sphere(0.2).mass(0.0));
    let arm = world.spawn(BodyDesc::cuboid([1.0, 0.05, 0.05]).position([1.0, 0.0, 0.0]));
    let motor = ConstraintMotor {
        target_velocity: 0.0,
        max_force: 20.0,
        target_position: Some(1.2),
        stiffness: 0.15,
        damping: 0.3,
    };
    let dofs = [
        DofDesc::driven(motor),
        DofDesc::free(),
        DofDesc::free(),
        DofDesc::locked(),
        DofDesc::locked(),
        DofDesc::locked(),
    ];
    let desc = ConstraintDesc::six_dof([0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        .dofs(dofs);
    world.add_constraint(anchor, arm, desc);
    for _ in 0..300 {
        world.step(DT);
    }
    world.wait();
    let state = world.read_state(arm);
    let reach =
        (state.position[0] * state.position[0] + state.position[2] * state.position[2]).sqrt();
    assert!(
        reach > 0.9,
        "servo driven arm must stay anchored, reach {reach}"
    );
}

#[test]
fn servo_drives_prismatic_to_target_distance() {
    let mut world = new_world(static_config());
    let anchor = world.spawn(BodyDesc::sphere(0.2).mass(0.0));
    let slider = world.spawn(BodyDesc::sphere(0.2).position([0.0, 0.0, 0.0]));
    world.add_constraint(
        anchor,
        slider,
        ConstraintDesc::prismatic([0.0; 3], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            .servo(2.0, 0.2, 0.4)
            .motor_force(30.0),
    );
    settle_until(&mut world, 300, |world| {
        (world.read_state(slider).position[0] - 2.0).abs() < 0.15
    });
    let x = world.read_state(slider).position[0];
    assert!(
        (x - 2.0).abs() < 0.15,
        "prismatic servo must approach the target distance, got {x}"
    );
}

#[test]
fn warm_start_off_keeps_constraint_stable() {
    let mut world = new_world(static_config());
    let pick = world.spawn(BodyDesc::sphere(0.2).mass(0.0));
    let load = world.spawn(BodyDesc::sphere(0.2).position([0.0, -2.0, 0.0]));
    world.add_constraint(
        pick,
        load,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 2.0),
    );
    for _i in 0..60 {
        world.step(DT);
    }
    world.wait();
    let hold = world.read_state(load).position[1];
    world.set_warm_start(world.constraints()[0], true);
    world.set_velocity(load, [0.0, -1.0, 0.0]);
    for _ in 0..120 {
        world.step(DT);
    }
    world.wait();
    assert!(
        world.read_state(load).position[1] < hold + 0.01,
        "distance constraint must keep holding the load"
    );
}

#[test]
fn updating_a_moved_constraint_edits_its_own_record() {
    let mut world = new_world(static_config());
    let anchor = world.spawn(BodyDesc::static_sphere(0.05));
    let loose = world.spawn(BodyDesc::sphere(0.05).position([1.0, 0.0, 0.0]));
    let far = world.spawn(BodyDesc::sphere(0.05));
    let dropped = world.add_constraint(
        anchor,
        loose,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0),
    );
    world.add_constraint(
        anchor,
        far,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 5.0),
    );
    let moved = world.add_constraint(
        loose,
        far,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 10.0),
    );
    world.remove_constraint(dropped);
    assert_eq!(world.constraints()[0], moved);
    world.update_constraint(moved, ConstraintDesc::distance([0.0; 3], [0.0; 3], 20.0));
    settle_until(&mut world, 480, |world| {
        let held = distance(
            world.read_state(loose).position,
            world.read_state(far).position,
        );
        (held - 20.0).abs() < 0.5
    });
    let reach = distance(
        world.read_state(anchor).position,
        world.read_state(far).position,
    );
    assert!(
        (reach - 5.0).abs() < 0.5,
        "the untouched anchor-far span must keep its rest length, got {reach}"
    );
}

fn velocity_motor(speed: f32, max_force: f32) -> ConstraintMotor {
    ConstraintMotor {
        target_velocity: speed,
        max_force,
        target_position: None,
        stiffness: 0.0,
        damping: 0.0,
    }
}

#[test]
fn pulley_patch_keeps_the_constraint_alive() {
    let mut world = new_world(static_config());
    let base = world.spawn(BodyDesc::sphere(0.1).mass(0.0));
    let first = world.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.2).position([2.0, 1.0, 0.0]));
    let fixed_a = [0.0f32, 3.0, 0.0];
    let fixed_b = [2.0f32, 3.0, 0.0];
    let length = 4.0;
    let joint = world.add_constraint(
        first,
        second,
        ConstraintDesc::pulley([0.0; 3], [0.0; 3], fixed_a, fixed_b, length),
    );
    let _ = base;
    world.set_motor(joint, 2.0, 50.0);
    world.set_limit(
        joint,
        Some(dynamis_model::ConstraintLimit {
            min: -1.0,
            max: 1.0,
        }),
    );
    world.set_spring(
        joint,
        Some(dynamis_model::ConstraintSpring {
            frequency: 2.0,
            damping_ratio: 0.5,
        }),
    );
    world.set_break_threshold(
        joint,
        Some(dynamis_model::ConstraintBreak {
            force: 0.0,
            torque: 0.0,
        }),
    );
    world.set_warm_start(joint, false);
    world.set_swing_limits(
        joint,
        Some(dynamis_model::ConstraintSwing {
            swing_a: 0.5,
            swing_b: 0.5,
        }),
    );
    world.set_constraint_disable_collisions(joint, false);
    assert!(
        world.constraints().contains(&joint),
        "patching a pulley must keep the constraint alive"
    );
    for _ in 0..90 {
        world.apply_force(first, [0.0, 6.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let rope = distance(world.read_state(first).position, fixed_a)
        + distance(world.read_state(second).position, fixed_b);
    assert!(
        (rope - length).abs() < 0.3,
        "a patched pulley must keep solving the rope length, got {rope}"
    );
}

#[test]
fn limited_dof_stays_driven_inside_its_limit() {
    let mut world = new_world(static_config());
    let base = world.spawn(BodyDesc::sphere(0.2).mass(0.0));
    let arm = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    world.add_constraint(
        base,
        arm,
        ConstraintDesc::six_dof([0.0; 3], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]).dofs(
            [
                DofDesc::free(),
                DofDesc::limited(-0.6, 0.6).motor(velocity_motor(3.0, 120.0)),
                DofDesc::free(),
                DofDesc::locked(),
                DofDesc::locked(),
                DofDesc::locked(),
            ],
        ),
    );
    for _ in 0..30 {
        world.step(DT);
    }
    world.wait();
    let travelled = world.read_state(arm).position[0];
    assert!(
        travelled < 0.9,
        "a limited dof must still follow its motor, got x {travelled}"
    );
    for _ in 0..90 {
        world.step(DT);
    }
    world.wait();
    let capped = world.read_state(arm).position[0];
    assert!(
        (0.3..0.5).contains(&capped),
        "the dof limit must stop the driven dof near 0.4, got x {capped}"
    );
}

#[test]
fn limited_angular_dof_stays_driven_inside_its_limit() {
    let mut world = new_world(static_config());
    let base = world.spawn(BodyDesc::sphere(0.2).mass(0.0));
    let arm = world.spawn(BodyDesc::sphere(0.2));
    world.add_constraint(
        base,
        arm,
        ConstraintDesc::six_dof([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]).dofs([
            DofDesc::locked(),
            DofDesc::locked(),
            DofDesc::locked(),
            DofDesc::limited(-0.6, 0.6).motor(velocity_motor(3.0, 120.0)),
            DofDesc::locked(),
            DofDesc::locked(),
        ]),
    );
    for _ in 0..120 {
        world.step(DT);
    }
    world.wait();
    let orientation = world.read_state(arm).orientation;
    let tilt = 2.0 * orientation[1].atan2(orientation[3]);
    assert!(
        tilt > 0.25,
        "a limited angular dof must still follow its motor, got tilt {tilt}"
    );
    assert!(
        tilt < 0.75,
        "the angular dof limit must stop the driven dof, got tilt {tilt}"
    );
}

#[test]
fn dof_patches_drive_and_cap_a_live_constraint() {
    let mut world = new_world(static_config());
    let base = world.spawn(BodyDesc::sphere(0.2).mass(0.0));
    let arm = world.spawn(BodyDesc::sphere(0.2));
    let joint = world.add_constraint(
        base,
        arm,
        ConstraintDesc::six_dof([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
            .dofs([DofDesc::locked(); 6]),
    );
    for _ in 0..10 {
        world.step(DT);
    }
    world.wait();
    assert!(
        world.read_state(arm).orientation[1].abs() < 1e-3,
        "a fully locked six dof must not rotate"
    );
    world.set_dof_locked(joint, 3, false);
    world.set_dof_motor(joint, 3, Some(velocity_motor(3.0, 120.0)));
    world.set_warm_start(joint, false);
    for _ in 0..60 {
        world.step(DT);
    }
    world.wait();
    let freed = world.read_state(arm).orientation[1].abs();
    assert!(
        freed > 0.05,
        "a live dof patch must start driving the arm, got {freed}"
    );
    world.set_dof_limit(
        joint,
        3,
        Some(dynamis_model::ConstraintLimit {
            min: -0.2,
            max: 0.2,
        }),
    );
    for _ in 0..120 {
        world.step(DT);
    }
    world.wait();
    let capped =
        2.0 * world.read_state(arm).orientation[1].atan2(world.read_state(arm).orientation[3]);
    assert!(
        capped < 0.4,
        "a live dof limit must cap the driven rotation, got {capped}"
    );
}
