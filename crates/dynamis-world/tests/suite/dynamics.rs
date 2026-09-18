use super::common::{
    DT, asleep, distance, gravity_config, observed_world, settle, settle_until, static_config,
    static_sphere_ground, symplectic_fall,
};
use dynamis_model::{BodyDesc, ConstraintDesc, PhysicsConfig};
use dynamis_world::World;

const GRAVITY: f32 = 9.81;

#[test]
fn free_fall_matches_closed_form() {
    let mut world = observed_world(gravity_config());
    let ball = world.spawn(
        BodyDesc::sphere(0.1)
            .position([0.0, 10.0, 0.0])
            .mass(1.0)
            .restitution(0.0),
    );
    const STEPS: u32 = 30;
    let substeps = world.config().substeps;
    for _ in 0..STEPS {
        world.step(DT);
    }
    world.wait();
    let state = world.read_state(ball);
    let expected_y = 10.0 + symplectic_fall(GRAVITY, STEPS, 0.0, substeps);
    assert!((state.position[1] - expected_y).abs() < 1e-3);
    let expected_v = -GRAVITY * DT * STEPS as f32;
    assert!((state.velocity[1] - expected_v).abs() < 1e-3);
}

#[test]
fn gravity_change_mid_flight_resumes_closed_form() {
    let mut world = observed_world(PhysicsConfig {
        gravity: [0.0, -4.0, 0.0],
        damping: 0.0,
        angular_damping: 0.0,
        ..PhysicsConfig::default()
    });
    let ball = world.spawn(BodyDesc::sphere(0.2).position([0.0, 10.0, 0.0]));
    const FIRST: u32 = 10;
    let substeps = world.config().substeps;
    for _ in 0..FIRST {
        world.step(DT);
    }
    world.set_gravity([0.0, -GRAVITY, 0.0]);
    const SECOND: u32 = 20;
    for _ in 0..SECOND {
        world.step(DT);
    }
    world.wait();
    let state = world.read_state(ball);
    let y_first = 10.0 + symplectic_fall(4.0, FIRST, 0.0, substeps);
    let v_first = -4.0 * DT * FIRST as f32;
    let expected = y_first + symplectic_fall(GRAVITY, SECOND, v_first, substeps);
    assert!(
        (state.position[1] - expected).abs() < 1e-3,
        "updated gravity must govern the remaining fall"
    );
}

#[test]
fn ball_rests_on_ground_and_stays_asleep() {
    let mut world = observed_world(gravity_config());
    static_sphere_ground(&mut world, 1.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 3.0, 0.0]));
    settle_until(&mut world, 90, |world| asleep(world));
    let rest_y = world.read_state(ball).position[1];
    assert!(
        (rest_y - 1.5).abs() < 0.05,
        "ball must rest on the sphere top, got {rest_y}"
    );
    assert!(world.read_state(ball).sleeping, "resting body must sleep");
    settle(&mut world, 30);
    assert!(
        (world.read_state(ball).position[1] - rest_y).abs() < 1e-3,
        "sleeping body must not drift"
    );
}

#[test]
fn overlapping_bodies_separate() {
    let mut world = observed_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.5).position([0.0, -0.4, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.4, 0.0]));
    settle(&mut world, 16);
    let separation = world.read_state(second).position[1] - world.read_state(first).position[1];
    assert!(separation >= 0.99, "bodies must separate, got {separation}");
}

#[test]
fn restitution_bounces_fast_and_calms_slow_impact() {
    let mut world = observed_world(static_config());
    static_sphere_ground(&mut world, 1.0);
    let bouncing = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 1.5, 0.0])
            .velocity([0.0, -2.0, 0.0])
            .restitution(0.8),
    );
    world.step(DT);
    world.wait();
    let bounce = world.read_state(bouncing);
    assert!(bounce.velocity[1] > 1.2, "fast impact must bounce upward");
    assert!(bounce.position[1] > 1.4);

    let mut calm = observed_world(static_config());
    static_sphere_ground(&mut calm, 1.0);
    let gentle = calm.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 1.5, 0.0])
            .velocity([0.0, -0.5, 0.0])
            .restitution(0.9),
    );
    calm.step(DT);
    calm.wait();
    let velocity = calm.read_state(gentle).velocity[1];
    assert!(
        velocity > -0.05 && velocity < 0.01,
        "threshold must suppress micro-bounce, got {velocity}"
    );
}

#[test]
fn friction_free_slides_then_grip_rolls() {
    let mut world = observed_world(gravity_config());
    static_sphere_ground(&mut world, 100.0);
    let ball = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 100.5, 0.0])
            .velocity([2.0, 0.0, 0.0])
            .friction(0.0),
    );
    settle(&mut world, 30);
    let slipping = world.read_state(ball);
    assert!(
        slipping.velocity[0] > 1.9,
        "frictionless ball must keep sliding"
    );
    assert!(
        slipping.angular_velocity[2].abs() < 0.01,
        "no spin without friction"
    );

    world.set_friction(ball, 0.8);
    settle_until(&mut world, 600, |world| {
        let state = world.read_state(ball);
        state.angular_velocity[2] < -1.5
            && (state.velocity[0] + state.angular_velocity[2] * 0.5).abs() < 0.05
    });
    let rolling = world.read_state(ball);
    assert!(
        rolling.angular_velocity[2] < -1.5,
        "friction must spin the ball"
    );
    assert!(
        (rolling.velocity[0] + rolling.angular_velocity[2] * 0.5).abs() < 0.05,
        "ball must roll without slipping, got slip {}",
        rolling.velocity[0] + rolling.angular_velocity[2] * 0.5
    );
}

#[test]
fn stacked_bodies_do_not_collapse() {
    let mut world = observed_world(gravity_config());
    static_sphere_ground(&mut world, 1.0);
    let lower = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.5, 0.0]));
    let upper = world.spawn(BodyDesc::sphere(0.5).position([0.0, 2.5, 0.0]));
    settle_until(&mut world, 120, |world| asleep(world));
    let lower_y = world.read_state(lower).position[1];
    let upper_y = world.read_state(upper).position[1];
    assert!(lower_y > 1.4 && lower_y < 1.52);
    assert!(upper_y > 2.4 && upper_y < 2.6);
}

#[test]
fn a_resting_tower_holds_its_spacing() {
    let mut world = observed_world(PhysicsConfig::default());
    world.spawn(
        BodyDesc::cuboid([40.0, 0.5, 40.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let mut tower = Vec::new();
    for level in 0..16 {
        tower.push(
            world.spawn(
                BodyDesc::cuboid([0.5; 3])
                    .position([0.0, 0.5 + level as f32 * 1.0005, 0.0])
                    .friction(0.7)
                    .restitution(0.0),
            ),
        );
    }
    settle(&mut world, 240);
    let margin = world.config().contact_margin;
    let slop = world.config().slop;
    for (level, handle) in tower.iter().enumerate() {
        let state = world.read_state(*handle);
        let expected = 0.5 + level as f32;
        assert!(
            (state.position[1] - expected).abs() <= margin + level as f32 * slop,
            "tower level {level} must hold its spacing: y={} expected={expected}",
            state.position[1]
        );
        assert!(
            state.position[0].abs() < 0.2 && state.position[2].abs() < 0.2,
            "tower level {level} must not wander: {:?}",
            state.position
        );
    }
}

#[test]
fn a_dropped_body_settles_within_the_contact_margin() {
    let margin = 0.02;
    let mut world = observed_world(PhysicsConfig {
        contact_margin: margin,
        ..PhysicsConfig::default()
    });
    world.spawn(
        BodyDesc::cuboid([40.0, 0.5, 40.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let box_body = world.spawn(
        BodyDesc::cuboid([0.5; 3])
            .position([0.0, 2.5, 0.0])
            .friction(0.7)
            .restitution(0.0),
    );
    settle(&mut world, 240);
    let state = world.read_state(box_body);
    let overlap = 0.5 - state.position[1];
    assert!(
        overlap <= margin + world.config().slop,
        "a landing body must not sink past the contact margin: overlap={overlap}"
    );
}

#[test]
fn ccd_flag_stops_bullet_that_would_tunnel() {
    let mut world = observed_world(static_config());
    static_sphere_ground(&mut world, 0.2);
    let bullet = world.spawn(
        BodyDesc::sphere(0.3)
            .position([-12.6, 0.0, 0.0])
            .velocity([90.0, 0.0, 0.0])
            .restitution(0.0),
    );
    settle(&mut world, 20);
    assert!(
        world.read_state(bullet).position[0] > 1.0,
        "unflagged bullet must tunnel through the small target, got x={}",
        world.read_state(bullet).position[0]
    );

    let mut guarded = observed_world(static_config());
    static_sphere_ground(&mut guarded, 0.2);
    let bullet = guarded.spawn(
        BodyDesc::sphere(0.3)
            .position([-12.6, 0.0, 0.0])
            .velocity([90.0, 0.0, 0.0])
            .restitution(0.0),
    );
    guarded.set_ccd(bullet, true);
    settle(&mut guarded, 20);
    let state = guarded.read_state(bullet);
    assert!(
        state.position[0] > -0.6 && state.position[0] < -0.45,
        "ccd bullet must stop at the target surface (-0.5), got x={}",
        state.position[0]
    );
}

#[test]
fn ccd_stops_an_elongated_body_by_its_leading_face() {
    let mut world = observed_world(static_config());
    let _wall = world.spawn(
        BodyDesc::cuboid([0.05, 5.0, 5.0])
            .mass(0.0)
            .position([0.0, 0.0, 0.0]),
    );
    let rod = world.spawn(
        BodyDesc::cuboid([1.0, 0.05, 0.05])
            .position([-4.0, 0.0, 0.0])
            .velocity([240.0, 0.0, 0.0])
            .restitution(0.0),
    );
    world.set_ccd(rod, true);
    world.step(DT);
    world.wait();
    let tip = world.read_state(rod).position[0] + 1.0;
    assert!(
        tip < 0.05,
        "ccd must sweep the rod's leading face before it crosses the wall, tip reached {tip}"
    );
    settle(&mut world, 20);
    let tip = world.read_state(rod).position[0] + 1.0;
    assert!(
        tip < 0.05,
        "a ccd rod must never cross the wall it swept, tip reached {tip}"
    );
    assert!(
        tip > -0.2,
        "a ccd rod must come to rest against the wall, tip reached {tip}"
    );
}

#[test]
fn ccd_keeps_a_spinning_rod_out_of_the_wall_it_sweeps() {
    use dynamis_model::math::quat_rotate;

    const WALL_FACE: f32 = 0.85;

    fn deepest_tip(ccd: bool) -> f32 {
        let mut world = observed_world(static_config());
        world.spawn(
            BodyDesc::cuboid([0.05, 2.0, 2.0])
                .mass(0.0)
                .position([0.9, 0.0, 0.0]),
        );
        let angle: f32 = std::f32::consts::PI / 3.0;
        let rod = world.spawn(
            BodyDesc::cuboid([1.0, 0.05, 0.05])
                .orientation([0.0, 0.0, (0.5 * angle).sin(), (0.5 * angle).cos()])
                .angular_velocity([0.0, 0.0, -120.0])
                .ccd(ccd),
        );
        let mut deepest = f32::MIN;
        for _ in 0..30 {
            world.step(DT);
            world.wait();
            let state = world.read_state(rod);
            let first = quat_rotate(state.orientation, [1.0, 0.0, 0.0]);
            let second = quat_rotate(state.orientation, [-1.0, 0.0, 0.0]);
            deepest =
                deepest.max((state.position[0] + first[0]).max(state.position[0] + second[0]));
        }
        deepest
    }

    let guarded = deepest_tip(true);
    assert!(
        guarded < WALL_FACE,
        "continuous collision must stop a spinning rod at the wall face, reached {guarded}"
    );
    assert!(
        guarded > WALL_FACE - 0.1,
        "a spinning rod under continuous collision must reach the wall, reached {guarded}"
    );
    let free = deepest_tip(false);
    assert!(
        free > guarded,
        "a rod without continuous collision must drive deeper into the wall, reached {free} against {guarded}"
    );
}

#[test]
fn ccd_bullet_stops_at_the_nearest_obstacle_of_a_chain() {
    let mut world = observed_world(static_config());
    let mut nearest = f32::MAX;
    for index in 0..81 {
        let x = -2.0 + index as f32 * 0.05;
        nearest = nearest.min(x);
        world.spawn(BodyDesc::static_sphere(0.15).position([x, 0.0, 0.0]));
    }
    let bullet = world.spawn(
        BodyDesc::sphere(0.3)
            .position([-12.6, 0.0, 0.0])
            .velocity([90.0, 0.0, 0.0])
            .restitution(0.0),
    );
    world.set_ccd(bullet, true);
    for _ in 0..20 {
        world.step(DT);
        world.wait();
        let x = world.read_state(bullet).position[0];
        assert!(
            x < nearest - 0.4,
            "ccd bullet must stop at the nearest obstacle surface ({}), got x={x}",
            nearest - 0.45
        );
    }
}

#[test]
fn kinematic_platform_carries_ball_and_ignores_gravity() {
    let mut world = observed_world(gravity_config());
    let platform = world.spawn(
        BodyDesc::cuboid([2.0, 0.2, 2.0])
            .position([0.0, 2.0, 0.0])
            .kinematic(true)
            .velocity([1.0, 0.0, 0.0]),
    );
    let ball = world.spawn(BodyDesc::sphere(0.3).position([0.0, 2.3, 0.0]));
    let floater = world.spawn(BodyDesc::sphere(0.5).kinematic(true));
    settle_until(&mut world, 120, |world| {
        world.read_state(platform).position[0] > 1.0
    });
    let platform_x = world.read_state(platform).position[0];
    assert!(platform_x > 0.5, "kinematic platform must advance");
    assert!(
        (world.read_state(ball).position[1] - 2.5).abs() < 0.5,
        "ball must ride the platform surface"
    );
    assert!(
        world.read_state(ball).position[0] > 0.01,
        "platform must push the ball along"
    );
    assert_eq!(
        world.read_state(floater).position[1],
        0.0,
        "kinematic body must ignore gravity"
    );
}

#[test]
fn idle_body_sleeps_and_impact_wakes() {
    let mut world = observed_world(gravity_config());
    let ground = world.spawn(BodyDesc::static_sphere(5.0).position([0.0, -1.0, 0.0]));
    let _ = ground;
    let target = world.spawn(BodyDesc::sphere(0.5).position([0.0, 4.5, 0.0]));
    settle_until(&mut world, 40, |world| asleep(world));
    assert!(
        world.read_state(target).sleeping,
        "resting target must sleep before impact"
    );
    let striker = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 12.0, 0.0])
            .velocity([0.0, -10.0, 0.0]),
    );
    settle_until(&mut world, 120, |world| !world.read_state(target).sleeping);
    assert!(
        !world.read_state(target).sleeping,
        "impact must wake the target"
    );
    let _ = striker;
}

#[test]
fn sleeping_body_wakes_and_shares_the_impact() {
    let mut world = observed_world(static_config());
    let target = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 0.0]));
    settle(&mut world, 40);
    assert!(world.read_state(target).sleeping);
    let striker = world.spawn(
        BodyDesc::sphere(0.5)
            .position([-5.0, 0.0, 0.0])
            .velocity([7.0, 0.0, 0.0]),
    );
    settle(&mut world, 40);
    let ahead = world.read_state(target);
    let behind = world.read_state(striker);
    assert!(!ahead.sleeping, "contact must wake the sleeper");
    assert!(
        behind.position[0] < ahead.position[0] - 0.98,
        "the striker must never pass through the sleeping body, gap={}",
        ahead.position[0] - behind.position[0]
    );
    assert!(
        (behind.velocity[0] - 3.5).abs() < 0.2 && (ahead.velocity[0] - 3.5).abs() < 0.2,
        "an inelastic impact must split the momentum between equal masses, striker={} target={}",
        behind.velocity[0],
        ahead.velocity[0]
    );
}

#[test]
fn sleep_and_wake_commands_toggle_state() {
    let mut world = observed_world(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5));
    settle(&mut world, 5);
    world.sleep(ball);
    settle(&mut world, 2);
    assert!(world.read_state(ball).sleeping, "sleep command must freeze");
    world.wake(ball);
    settle(&mut world, 2);
    assert!(!world.read_state(ball).sleeping, "wake command must revive");
}

#[test]
fn patches_and_impulses_wake_sleeping_body() {
    let mut world = observed_world(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5));
    settle_until(&mut world, 40, |world| asleep(world));
    assert!(world.read_state(ball).sleeping);
    world.set_position(ball, [2.0, 0.0, 0.0]);
    settle(&mut world, 2);
    assert!(!world.read_state(ball).sleeping, "position patch must wake");
    settle_until(&mut world, 40, |world| asleep(world));
    assert!(world.read_state(ball).sleeping);
    world.apply_impulse(ball, [0.0, 0.0, 1.0]);
    settle(&mut world, 2);
    assert!(!world.read_state(ball).sleeping, "impulse must wake");
}

#[test]
fn constraint_linked_bodies_sleep_and_wake_together() {
    let mut world = observed_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.3));
    let second = world.spawn(BodyDesc::sphere(0.3).position([2.0, 0.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 2.0),
    );
    settle_until(&mut world, 50, |world| asleep(world));
    assert!(
        world.read_state(first).sleeping && world.read_state(second).sleeping,
        "linked bodies must share the sleep state"
    );
    world.wake(first);
    settle(&mut world, 3);
    assert!(
        !world.read_state(first).sleeping && !world.read_state(second).sleeping,
        "wake must propagate through the constraint island"
    );
}

#[test]
fn distant_idle_bodies_sleep_independently() {
    let mut world = observed_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.3));
    let second = world.spawn(BodyDesc::sphere(0.3).position([50.0, 0.0, 0.0]));
    settle_until(&mut world, 50, |world| asleep(world));
    assert!(world.read_state(first).sleeping && world.read_state(second).sleeping);
    world.wake(first);
    settle(&mut world, 3);
    assert!(
        !world.read_state(first).sleeping && world.read_state(second).sleeping,
        "unconnected bodies must not propagate wake"
    );
}

#[test]
fn driven_contact_island_stays_awake() {
    let mut world = observed_world(gravity_config());
    let _ground = world.spawn(BodyDesc::static_sphere(5.0).position([0.0, -1.0, 0.0]));
    let lower = world.spawn(BodyDesc::sphere(0.5).position([0.0, 4.5, 0.0]));
    let upper = world.spawn(BodyDesc::sphere(0.5).position([0.0, 5.3, 0.0]));
    settle_until(&mut world, 50, |world| asleep(world));
    assert!(world.read_state(lower).sleeping && world.read_state(upper).sleeping);
    for _ in 0..20 {
        world.apply_force(upper, [3.0, 0.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    assert!(
        !world.read_state(lower).sleeping && !world.read_state(upper).sleeping,
        "driven contact island must stay awake"
    );
    assert!(world.read_state(upper).position[0] > 0.05);
}

#[test]
fn resting_contact_carries_warm_start_impulse() {
    let mut world = observed_world(gravity_config());
    let _ground = world.spawn(BodyDesc::static_sphere(5.0).position([0.0, -1.0, 0.0]));
    let _ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 4.5, 0.0]));
    settle_until(&mut world, 60, |world| asleep(world));
    let resting = world.inspect_contacts();
    assert_eq!(
        resting.len(),
        1,
        "resting ball must hold exactly one contact"
    );
    let impulse = resting[0].points[0].normal_impulse;
    assert!(
        impulse > 0.01,
        "resting contact must accumulate normal impulse, got {impulse}"
    );
    for point in &resting[0].points {
        assert!(point.normal_impulse.is_finite() && point.normal_impulse >= 0.0);
    }
    let pair = (resting[0].first, resting[0].second);
    world.step(DT);
    let relayed = world.inspect_contacts();
    assert_eq!(
        relayed.len(),
        1,
        "the archived contact must be matched again"
    );
    assert_eq!(
        (relayed[0].first, relayed[0].second),
        pair,
        "the same pair must keep its contact"
    );
    assert!(
        relayed[0].points[0].normal_impulse > 0.01,
        "archived contact must relay the solved impulse into the next frame"
    );
}

#[test]
fn settled_stack_drifts_nothing_across_frames() {
    let mut world = observed_world(gravity_config());
    let _ground = world.spawn(BodyDesc::static_sphere(5.0).position([0.0, -1.0, 0.0]));
    let mut stack = Vec::new();
    for index in 0..3 {
        stack.push(
            world.spawn(
                BodyDesc::sphere(0.4)
                    .position([0.0, 4.7 + index as f32 * 0.81, 0.0])
                    .restitution(0.0),
            ),
        );
    }
    let snapshot = |world: &mut World| {
        stack
            .iter()
            .map(|handle| world.read_state(*handle).position)
            .collect::<Vec<_>>()
    };
    let mut previous = snapshot(&mut world);
    settle_until(&mut world, 180, |world| {
        let current = snapshot(world);
        let settled = current
            .iter()
            .zip(&previous)
            .all(|(now, before)| distance(*now, *before) < 1e-5);
        previous = current;
        settled
    });
    let baseline: Vec<_> = stack
        .iter()
        .map(|body| world.read_state(*body).position[1])
        .collect();
    settle(&mut world, 30);
    for (index, body) in stack.iter().enumerate() {
        let drift = (world.read_state(*body).position[1] - baseline[index]).abs();
        assert!(
            drift < 1e-3,
            "settled stack body {index} must stay still, drifted {drift}"
        );
    }
}

#[test]
fn kinematic_capsule_overlap_pushes_dynamic_box() {
    let mut world = super::common::observed_world(super::common::gravity_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let box_body = world.spawn(
        BodyDesc::cuboid([0.3, 0.3, 0.3])
            .position([1.6, 0.8, 0.0])
            .friction(0.6),
    );
    let pusher = world.spawn(
        BodyDesc::new(dynamis_model::ColliderDesc::new(
            dynamis_model::Shape::capsule(0.4, 0.5),
        ))
        .position([1.25, 0.8, 0.0])
        .kinematic(true)
        .velocity([1.0, 0.0, 0.0]),
    );
    let _ = pusher;
    for _ in 0..60 {
        world.step(DT);
    }
    world.wait();
    let state = world.read_state(box_body);
    assert!(
        state.position[0] > 1.75,
        "overlapping kinematic capsule must push the box, got x={}",
        state.position[0]
    );
}

#[test]
fn a_sphere_swallowed_by_a_flat_box_escapes_through_the_nearest_face() {
    let mut world = observed_world(super::common::gravity_config());
    world.spawn(BodyDesc::cuboid([2.0, 0.5, 2.0]).mass(0.0));
    let ball = world.spawn(BodyDesc::sphere(0.3).position([0.0, 0.2, 0.0]));
    settle_until(&mut world, 240, |world| asleep(world));
    let state = world.read_state(ball);
    assert!(
        state.position[1] > 0.75,
        "the swallowed sphere must pop out of the top face, got {:?}",
        state.position
    );
    assert!(
        state.position[0].abs() < 0.2 && state.position[2].abs() < 0.2,
        "the sphere must not squirt sideways, got {:?}",
        state.position
    );
}

#[test]
fn clearing_the_ccd_flag_restores_tunneling() {
    let mut world = observed_world(static_config());
    static_sphere_ground(&mut world, 0.2);
    let bullet = world.spawn(
        BodyDesc::sphere(0.3)
            .position([-12.6, 0.0, 0.0])
            .velocity([90.0, 0.0, 0.0])
            .restitution(0.0)
            .ccd(true),
    );
    world.set_ccd(bullet, false);
    settle(&mut world, 20);
    assert!(
        world.read_state(bullet).position[0] > 1.0,
        "clearing the ccd flag must restore tunneling, got x={}",
        world.read_state(bullet).position[0]
    );
}

#[test]
fn a_leaning_tower_holds_through_friction() {
    let mut world = observed_world(PhysicsConfig::default());
    world.spawn(
        BodyDesc::cuboid([40.0, 0.5, 40.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let lean = 0.1;
    let mut tower = Vec::new();
    for level in 0..6 {
        tower.push(
            world.spawn(
                BodyDesc::cuboid([0.5; 3])
                    .position([level as f32 * lean, 0.5 + level as f32, 0.0])
                    .friction(0.5)
                    .restitution(0.0),
            ),
        );
    }
    settle(&mut world, 480);
    for (level, handle) in tower.iter().enumerate() {
        let state = world.read_state(*handle);
        let expected = level as f32 * lean;
        assert!(
            (state.position[0] - expected).abs() < 0.15,
            "tower level {level} must hold its lean: x={} expected={expected}",
            state.position[0]
        );
        assert!(
            (state.position[1] - (0.5 + level as f32)).abs() < 0.1,
            "tower level {level} must hold its spacing: y={}",
            state.position[1]
        );
    }
}

#[test]
fn a_ball_rests_on_the_face_of_a_floor_far_wider_than_it_is_deep() {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([60.0, 0.5, 60.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let ball = world.spawn(BodyDesc::sphere(0.3).position([20.0, 1.0, 20.0]));
    settle_until(&mut world, 240, |world| asleep(world));
    let state = world.read_state(ball);
    assert!(
        (state.position[1] - 0.3).abs() < 0.05,
        "a ball must rest on the face of a large floor, got {:?}",
        state.position
    );
}
