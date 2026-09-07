use super::common::{
    DT, gravity_config, read_records, settle, sim, static_config, static_sphere_ground,
};
use dynamis_layout::ContactRecord;
use dynamis_model::{BodyDesc, ConstraintDesc, PhysicsConfig};
use dynamis_sim::{DebugBuffer, Simulation};

const GRAVITY: f32 = 9.81;

#[test]
fn free_fall_matches_closed_form() {
    let mut world = sim(4, gravity_config());
    let ball = world.spawn(
        BodyDesc::sphere(0.1)
            .position([0.0, 10.0, 0.0])
            .mass(1.0)
            .restitution(0.0),
    );
    const STEPS: u32 = 30;
    for _ in 0..STEPS {
        world.step(DT);
    }
    world.wait();
    let state = world.read_state(ball);
    let expected_y = 10.0 - 0.5 * GRAVITY * DT * DT * (STEPS as f32 * (STEPS as f32 + 1.0));
    assert!((state.position[1] - expected_y).abs() < 1e-3);
    let expected_v = -GRAVITY * DT * STEPS as f32;
    assert!((state.velocity[1] - expected_v).abs() < 1e-3);
}

#[test]
fn gravity_change_mid_flight_resumes_closed_form() {
    let mut world = sim(
        4,
        PhysicsConfig {
            gravity: [0.0, -4.0, 0.0],
            damping: 0.0,
            angular_damping: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let ball = world.spawn(BodyDesc::sphere(0.2).position([0.0, 10.0, 0.0]));
    const FIRST: u32 = 10;
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
    let y_first = 10.0 - 0.5 * 4.0 * DT * DT * (FIRST as f32 * (FIRST as f32 + 1.0));
    let v_first = -4.0 * DT * FIRST as f32;
    let expected = y_first + v_first * DT * SECOND as f32
        - 0.5 * GRAVITY * DT * DT * (SECOND as f32 * (SECOND as f32 + 1.0));
    assert!(
        (state.position[1] - expected).abs() < 1e-3,
        "updated gravity must govern the remaining fall"
    );
}

#[test]
fn ball_rests_on_ground_and_stays_asleep() {
    let mut world = sim(4, gravity_config());
    static_sphere_ground(&mut world, 1.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 3.0, 0.0]));
    settle(&mut world, 90);
    let rest_y = world.read_state(ball).position[1];
    assert!(
        (rest_y - 1.5).abs() < 0.05,
        "ball must rest on the sphere top, got {rest_y}"
    );
    assert!(world.read_state(ball).sleeping, "resting body must sleep");
    settle(&mut world, 60);
    assert!(
        (world.read_state(ball).position[1] - rest_y).abs() < 1e-3,
        "sleeping body must not drift"
    );
}

#[test]
fn overlapping_bodies_separate() {
    let mut world = sim(4, static_config());
    let first = world.spawn(BodyDesc::sphere(0.5).position([0.0, -0.4, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.4, 0.0]));
    settle(&mut world, 16);
    let separation = world.read_state(second).position[1] - world.read_state(first).position[1];
    assert!(separation >= 0.99, "bodies must separate, got {separation}");
}

#[test]
fn restitution_bounces_fast_and_calms_slow_impact() {
    let mut world = sim(4, static_config());
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

    let mut calm = sim(4, static_config());
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
    let mut world = sim(4, gravity_config());
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
    settle(&mut world, 180);
    let rolling = world.read_state(ball);
    assert!(
        rolling.angular_velocity[2] < -1.5,
        "friction must spin the ball"
    );
    assert!(
        (rolling.velocity[0] + rolling.angular_velocity[2] * 0.5).abs() < 0.1,
        "ball must roll without slipping"
    );
}

#[test]
fn stacked_bodies_do_not_collapse() {
    let mut world = sim(8, gravity_config());
    static_sphere_ground(&mut world, 1.0);
    let lower = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.5, 0.0]));
    let upper = world.spawn(BodyDesc::sphere(0.5).position([0.0, 2.5, 0.0]));
    settle(&mut world, 120);
    let lower_y = world.read_state(lower).position[1];
    let upper_y = world.read_state(upper).position[1];
    assert!(lower_y > 1.4 && lower_y < 1.52);
    assert!(upper_y > 2.4 && upper_y < 2.6);
}

#[test]
fn ccd_flag_stops_bullet_that_would_tunnel() {
    let mut world = sim(8, static_config());
    static_sphere_ground(&mut world, 0.2);
    let bullet = world.spawn(
        BodyDesc::sphere(0.3)
            .position([-12.0, 0.0, 0.0])
            .velocity([90.0, 0.0, 0.0])
            .restitution(0.0),
    );
    settle(&mut world, 20);
    assert!(
        world.read_state(bullet).position[0] > 1.0,
        "unflagged bullet must tunnel through the small target"
    );

    let mut guarded = sim(8, static_config());
    static_sphere_ground(&mut guarded, 0.2);
    let bullet = guarded.spawn(
        BodyDesc::sphere(0.3)
            .position([-12.0, 0.0, 0.0])
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
    assert!(
        state.velocity[0].abs() < 0.05,
        "ccd bullet must lose its impact velocity"
    );
}

#[test]
fn kinematic_platform_carries_ball_and_ignores_gravity() {
    let mut world = sim(8, gravity_config());
    let platform = world.spawn(
        BodyDesc::cuboid([2.0, 0.2, 2.0])
            .position([0.0, 2.0, 0.0])
            .kinematic(true)
            .velocity([1.0, 0.0, 0.0]),
    );
    let ball = world.spawn(BodyDesc::sphere(0.3).position([0.0, 2.3, 0.0]));
    let floater = world.spawn(BodyDesc::sphere(0.5).kinematic(true));
    settle(&mut world, 120);
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
    let mut world = sim(8, gravity_config());
    let ground = world.spawn(BodyDesc::static_sphere(5.0).position([0.0, -1.0, 0.0]));
    let _ = ground;
    let target = world.spawn(BodyDesc::sphere(0.5).position([0.0, 4.5, 0.0]));
    settle(&mut world, 40);
    assert!(
        world.read_state(target).sleeping,
        "resting target must sleep before impact"
    );
    let striker = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 12.0, 0.0])
            .velocity([0.0, -10.0, 0.0]),
    );
    settle(&mut world, 60);
    assert!(
        !world.read_state(target).sleeping,
        "impact must wake the target"
    );
    let _ = striker;
}

#[test]
fn sleeping_body_blocks_until_impact_wakes_it() {
    let mut world = sim(8, static_config());
    let target = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 0.0]));
    settle(&mut world, 40);
    assert!(world.read_state(target).sleeping);
    let striker = world.spawn(
        BodyDesc::sphere(0.5)
            .position([-5.0, 0.0, 0.0])
            .velocity([7.0, 0.0, 0.0]),
    );
    settle(&mut world, 40);
    assert!(
        world.read_state(striker).position[0] < -0.9,
        "striker must not pass through the sleeping body"
    );
    assert!(
        !world.read_state(target).sleeping,
        "contact must wake the sleeper"
    );
}

#[test]
fn sleep_and_wake_commands_toggle_state() {
    let mut world = sim(4, static_config());
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
    let mut world = sim(8, static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5));
    settle(&mut world, 40);
    assert!(world.read_state(ball).sleeping);
    world.set_position(ball, [2.0, 0.0, 0.0]);
    settle(&mut world, 2);
    assert!(!world.read_state(ball).sleeping, "position patch must wake");
    settle(&mut world, 40);
    assert!(world.read_state(ball).sleeping);
    world.apply_impulse(ball, [0.0, 0.0, 1.0]);
    settle(&mut world, 2);
    assert!(!world.read_state(ball).sleeping, "impulse must wake");
}

#[test]
fn constraint_linked_bodies_sleep_and_wake_together() {
    let mut world = sim(8, static_config());
    let first = world.spawn(BodyDesc::sphere(0.3));
    let second = world.spawn(BodyDesc::sphere(0.3).position([2.0, 0.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 2.0),
    );
    settle(&mut world, 50);
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
    let mut world = sim(8, static_config());
    let first = world.spawn(BodyDesc::sphere(0.3));
    let second = world.spawn(BodyDesc::sphere(0.3).position([50.0, 0.0, 0.0]));
    settle(&mut world, 50);
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
    let mut world = sim(16, gravity_config());
    let _ground = world.spawn(BodyDesc::static_sphere(5.0).position([0.0, -1.0, 0.0]));
    let lower = world.spawn(BodyDesc::sphere(0.5).position([0.0, 4.5, 0.0]));
    let upper = world.spawn(BodyDesc::sphere(0.5).position([0.0, 5.3, 0.0]));
    settle(&mut world, 50);
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
    let mut world = sim(8, gravity_config());
    let _ground = world.spawn(BodyDesc::static_sphere(5.0).position([0.0, -1.0, 0.0]));
    let _ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 4.5, 0.0]));
    for _ in 0..60 {
        world.step(DT);
    }
    world.wait();
    let count = read_u32(&world, world.debug_buffer(DebugBuffer::ContactCount));
    assert_eq!(count, 1, "resting ball must hold exactly one contact");
    let contacts: Vec<ContactRecord> = read_records(
        &world,
        world.debug_buffer(DebugBuffer::Contacts),
        count as usize,
    );
    let impulse = contacts[0].points[0].accumulated_normal;
    assert!(
        impulse > 0.01,
        "resting contact must accumulate normal impulse, got {impulse}"
    );
    for point in &contacts[0].points[..contacts[0].point_count as usize] {
        assert!(point.accumulated_normal.is_finite() && point.accumulated_normal >= 0.0);
    }
    world.step(DT);
    world.wait();
    let archived: Vec<ContactRecord> = read_records(
        &world,
        world.debug_buffer(DebugBuffer::PrevContacts),
        count as usize,
    );
    assert_eq!(archived[0].a, contacts[0].a);
    assert_eq!(archived[0].b, contacts[0].b);
    assert_eq!(
        archived[0].points[0].accumulated_normal, impulse,
        "archived contact must relay the solved impulse into the next frame"
    );
}

#[test]
fn settled_stack_drifts_nothing_across_frames() {
    let mut world = sim(16, gravity_config());
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
    settle(&mut world, 180);
    let baseline: Vec<_> = stack
        .iter()
        .map(|body| world.read_state(*body).position[1])
        .collect();
    settle(&mut world, 60);
    for (index, body) in stack.iter().enumerate() {
        let drift = (world.read_state(*body).position[1] - baseline[index]).abs();
        assert!(
            drift < 1e-3,
            "settled stack body {index} must stay still, drifted {drift}"
        );
    }
}

fn read_u32(world: &Simulation, buffer: &wgpu::Buffer) -> u32 {
    read_records::<u32>(world, buffer, 3)[0]
}
