use super::common::{DT, gravity_config, observed_world, refused, settle, static_config};
use dynamis_abi::{COUNTER_IMPACTS, COUNTER_REFUSED_IMPACTS};
use dynamis_model::{BodyDesc, ColliderDesc, Shape};

fn armed_ground(world: &mut dynamis_world::World, threshold: f32) -> dynamis_model::BodyHandle {
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::cuboid([20.0, 0.5, 20.0])).impact(threshold))
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    )
}

#[test]
fn a_landing_body_reports_the_impulse_it_delivers() {
    let mut world = observed_world(static_config());
    armed_ground(&mut world, 1.0);
    let mass = 2.0;
    let speed = 3.0;
    let ball = world.spawn(
        BodyDesc::sphere(0.5)
            .mass(mass)
            .position([0.0, 0.49, 0.0])
            .velocity([0.0, -speed, 0.0]),
    );
    world.step(DT);
    world.wait();

    let impacts = world.drain_impacts();
    assert_eq!(impacts.len(), 1, "a landing delivers one impact");
    let impact = impacts[0];
    assert!(
        (impact.first == ball || impact.second == ball),
        "the impact must name the landing body"
    );
    let delivered = mass * speed;
    assert!(
        (impact.impulse - delivered).abs() < delivered * 0.1,
        "a {mass} kg body landing at {speed} m/s must deliver {delivered} N s, got {}",
        impact.impulse,
    );
    assert!(
        impact.normal[1] > 0.9,
        "the impact normal must face the ground, got {:?}",
        impact.normal
    );
    assert_eq!(
        impact.step, 0,
        "the impact belongs to the step it landed in"
    );
}

#[test]
fn a_silent_collider_never_reports() {
    let mut world = observed_world(static_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 0.49, 0.0])
            .velocity([0.0, -3.0, 0.0]),
    );
    world.step(DT);
    world.wait();
    assert!(
        world.drain_impacts().is_empty(),
        "a world that arms no impact threshold must report nothing"
    );
    assert_eq!(
        world.measured()[COUNTER_IMPACTS],
        0,
        "a world that arms no impact threshold must not count an impact"
    );
}

#[test]
fn either_collider_of_a_pair_arms_the_report() {
    let mut world = observed_world(static_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).impact(1000.0))
            .position([0.0, 0.49, 0.0])
            .velocity([0.0, -3.0, 0.0]),
    );
    world.step(DT);
    world.wait();
    assert!(
        world.drain_impacts().is_empty(),
        "the strict threshold of the armed collider must hold"
    );

    let mut world = observed_world(static_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).impact(1.0))
            .position([0.0, 0.49, 0.0])
            .velocity([0.0, -3.0, 0.0]),
    );
    world.step(DT);
    world.wait();
    assert_eq!(
        world.drain_impacts().len(),
        1,
        "an armed collider must report against a silent one"
    );
}

#[test]
fn a_gentle_touch_below_the_threshold_stays_silent() {
    let gentle = |threshold: f32| {
        let mut world = observed_world(static_config());
        armed_ground(&mut world, threshold);
        world.spawn(
            BodyDesc::sphere(0.5)
                .mass(1.0)
                .position([0.0, 0.49, 0.0])
                .velocity([0.0, -0.1, 0.0]),
        );
        world.step(DT);
        world.wait();
        world.drain_impacts().len()
    };
    assert_eq!(
        gentle(1000.0),
        0,
        "an impulse weaker than the armed force must not report"
    );
    assert_eq!(
        gentle(1.0),
        1,
        "the same touch must report once a collider arms a lower force"
    );
}

#[test]
fn a_sustained_load_reports_its_impact_once() {
    let mut world = observed_world(gravity_config());
    armed_ground(&mut world, 1.0);
    world.spawn(BodyDesc::sphere(0.5).mass(2.0).position([0.0, 1.5, 0.0]));

    let mut touches = 0;
    let mut landing = 0.0f32;
    settle(&mut world, 240);
    for _ in 0..240 {
        world.step(DT);
        world.wait();
        for impact in world.collect_impacts() {
            touches += 1;
            landing = landing.max(impact.impulse);
        }
    }
    assert_eq!(
        touches, 1,
        "a body that lands once must report one impact, got {touches} with a strongest {landing} N s"
    );
    assert!(
        landing > 2.0 * 9.81 * DT,
        "the landing must report the fall it carried, got {landing} N s"
    );
}

#[test]
fn a_slept_body_stays_silent() {
    let mut world = observed_world(gravity_config());
    armed_ground(&mut world, 1.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.5, 0.0]));
    settle(&mut world, 400);
    assert!(world.read_state(ball).sleeping, "the ball must sleep");
    world.collect_impacts();

    for _ in 0..30 {
        world.step(DT);
        world.wait();
        assert!(
            world.drain_impacts().is_empty(),
            "a sleeping body must not report an impact"
        );
    }
}

#[test]
fn a_sensor_reports_no_impact() {
    let mut world = observed_world(static_config());
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::cuboid([20.0, 0.5, 20.0])).sensor(true)).mass(0.0),
    );
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).impact(1.0))
            .position([0.0, 0.0, 0.0])
            .velocity([0.0, -3.0, 0.0]),
    );
    world.step(DT);
    world.wait();
    assert!(
        world.drain_impacts().is_empty(),
        "an unsolved sensor must not report an impact"
    );
}

#[test]
fn an_impact_flood_widens_the_stream_before_it_spills() {
    let mut world = observed_world(static_config());
    let floor = world.stream_capacity().rigid.impacts;
    let side = 7.0f32;
    let center = side * 0.7 * 0.5;
    for index in 0..343 {
        let position = [
            (index % 7) as f32 * 0.7,
            (index / 7 % 7) as f32 * 0.7,
            (index / 49) as f32 * 0.7,
        ];
        let inward = [
            center - position[0],
            center - position[1],
            center - position[2],
        ];
        let reach = (inward[0] * inward[0] + inward[1] * inward[1] + inward[2] * inward[2]).sqrt();
        let speed = if reach > 1e-3 { 20.0 / reach } else { 0.0 };
        world.spawn(
            BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).impact(0.0))
                .position(position)
                .velocity([inward[0] * speed, inward[1] * speed, inward[2] * speed]),
        );
    }
    world.step(DT);
    world.wait();
    let reported = world.drain_impacts().len() as u32;
    assert!(
        reported > 2 * 343,
        "a collapsing cluster must report every pair that closes, got {reported}"
    );
    assert_eq!(
        refused(&world, COUNTER_REFUSED_IMPACTS),
        0,
        "a freshly spawned flood must be served by its reservation"
    );

    world.step(DT);
    world.wait();
    assert!(
        world.stream_capacity().rigid.impacts > floor,
        "a flood must widen the impact stream"
    );
}

#[test]
fn an_armed_cluster_reports_its_collisions_and_its_removal() {
    let mut world = observed_world(static_config());
    let ground = armed_ground(&mut world, 1.0);
    let ball = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 0.49, 0.0])
            .velocity([0.0, -3.0, 0.0]),
    );
    world.step(DT);
    world.wait();
    assert_eq!(world.drain_impacts().len(), 1, "the landing must report");

    world.set_collider(
        ball,
        0,
        ColliderDesc::new(Shape::sphere(0.5)).impact(100000.0),
    );
    world.set_velocity(ball, [0.0, -3.0, 0.0]);
    world.step(DT);
    world.wait();
    assert!(
        world.drain_impacts().is_empty(),
        "the replaced threshold must silence the pair"
    );

    world.remove(ground);
    world.set_velocity(ball, [0.0, -3.0, 0.0]);
    world.step(DT);
    world.wait();
    assert!(
        world.drain_impacts().is_empty(),
        "a removed collider must arm nothing"
    );
}
