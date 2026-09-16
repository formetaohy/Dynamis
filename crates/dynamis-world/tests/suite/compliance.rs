use super::common::{gravity_config, observed_world, settle};
use dynamis_model::{BodyDesc, SurfaceDesc};
use dynamis_world::World;

const FREQUENCY: f32 = 10.0;

fn relaxation(frequency: f32) -> f32 {
    1.0 / (std::f32::consts::TAU * frequency)
}

fn relaxation_depth(left: f32, right: f32) -> f32 {
    9.81 * (relaxation(left) + relaxation(right)).powi(2)
}

fn soft_floor(world: &mut World, frequency: f32) {
    world.spawn(
        BodyDesc::cuboid([10.0, 0.5, 10.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0])
            .contact_frequency(frequency)
            .contact_damping_ratio(0.0),
    );
}

#[test]
fn a_soft_contact_rests_at_its_relaxation_depth() {
    let mut world = observed_world(gravity_config());
    soft_floor(&mut world, FREQUENCY);
    let sphere = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 0.5, 0.0])
            .contact_frequency(FREQUENCY)
            .contact_damping_ratio(0.0),
    );
    settle(&mut world, 240);
    let depth = 0.5 - world.read_state(sphere).position[1];
    let expected = relaxation_depth(FREQUENCY, FREQUENCY);
    assert!(
        (depth - expected).abs() < 0.1 * expected,
        "a soft contact must settle at its relaxation depth {expected}, got {depth}"
    );
}

#[test]
fn a_critically_damped_contact_holds_the_same_depth() {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([10.0, 0.5, 10.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0])
            .contact_frequency(FREQUENCY),
    );
    let sphere = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 0.5, 0.0])
            .contact_frequency(FREQUENCY),
    );
    settle(&mut world, 240);
    let depth = 0.5 - world.read_state(sphere).position[1];
    let expected = relaxation_depth(FREQUENCY, FREQUENCY);
    assert!(
        (depth - expected).abs() < 0.25 * expected,
        "a damped soft contact must hold its relaxation depth {expected}, got {depth}"
    );
}

#[test]
fn a_soft_contact_sinks_the_same_depth_under_any_mass() {
    let expected = relaxation_depth(FREQUENCY, FREQUENCY);
    for mass in [0.2f32, 5.0, 200.0] {
        let mut world = observed_world(gravity_config());
        soft_floor(&mut world, FREQUENCY);
        let sphere = world.spawn(
            BodyDesc::sphere(0.5)
                .mass(mass)
                .position([0.0, 0.5, 0.0])
                .contact_frequency(FREQUENCY)
                .contact_damping_ratio(0.0),
        );
        settle(&mut world, 240);
        let depth = 0.5 - world.read_state(sphere).position[1];
        assert!(
            (depth - expected).abs() < 0.12 * expected,
            "mass {mass} must rest at {expected}, got {depth}"
        );
    }
}

#[test]
fn a_stacked_pair_compresses_each_soft_contact_equally() {
    let mut world = observed_world(gravity_config());
    let single = relaxation_depth(FREQUENCY, FREQUENCY);
    soft_floor(&mut world, FREQUENCY);
    let lower = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 0.5, 0.0])
            .contact_frequency(FREQUENCY)
            .contact_damping_ratio(0.0),
    );
    let upper = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 1.5, 0.0])
            .contact_frequency(FREQUENCY)
            .contact_damping_ratio(0.0),
    );
    settle(&mut world, 300);
    let lower_y = world.read_state(lower).position[1];
    let upper_y = world.read_state(upper).position[1];
    let carried = 0.5 - lower_y;
    let carrying = 1.0 + lower_y - upper_y;
    assert!(
        (carried - 2.0 * single).abs() < 0.2 * single,
        "the loaded contact must compress by twice the single load, got {carried} against {single}"
    );
    assert!(
        (carrying - 2.0 * single).abs() < 0.2 * single,
        "the carrying contact must compress by its effective mass, got {carrying} against {single}"
    );
}

#[test]
fn contact_softness_combines_in_series() {
    let mut world = observed_world(gravity_config());
    soft_floor(&mut world, FREQUENCY);
    world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 0.5, 0.0])
            .contact_frequency(4.0 * FREQUENCY),
    );
    settle(&mut world, 120);
    let contacts = world.inspect_contacts();
    assert_eq!(contacts.len(), 1, "the pair must hold exactly one contact");
    let frequency = contacts[0].material.contact_frequency;
    let expected = 1.0 / (1.0 / FREQUENCY + 1.0 / (4.0 * FREQUENCY));
    assert!(
        (frequency - expected).abs() < 1.0e-4 * expected,
        "two soft surfaces must relax in series {expected}, got {frequency}"
    );
}

#[test]
fn a_rigid_contact_never_softens() {
    let mut world = observed_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([10.0, 0.5, 10.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let sphere = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.5, 0.0]));
    settle(&mut world, 120);
    let contacts = world.inspect_contacts();
    assert!(!contacts[0].material.contact_frequency.is_finite());
    assert!(0.5 - world.read_state(sphere).position[1] < 1.0e-3);
    assert_eq!(SurfaceDesc::new().contact_frequency, f32::INFINITY);
}
