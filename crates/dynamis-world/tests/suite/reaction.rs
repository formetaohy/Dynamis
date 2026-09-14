use super::common::{DT, gravity_config, new_world, static_config};
use dynamis_model::{BodyDesc, BodyHandle, ConstraintDesc, PhysicsConfig};
use dynamis_world::World;

fn hanging_load(world: &mut World, mass: f32, length: f32) -> (BodyHandle, BodyHandle) {
    let anchor = world.spawn(BodyDesc::static_sphere(0.1).position([0.0, 5.0, 0.0]));
    let load = world.spawn(
        BodyDesc::sphere(0.2)
            .mass(mass)
            .position([0.0, 5.0 - length, 0.0]),
    );
    world.add_constraint(
        anchor,
        load,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], length),
    );
    (anchor, load)
}

fn reaction_of(world: &mut World, constraint: dynamis_model::ConstraintHandle) -> [f32; 3] {
    world.inspect_constraint_force(constraint).force_on_second
}

#[test]
fn a_hanging_distance_joint_reports_the_weight_it_carries() {
    let mut world = new_world(gravity_config());
    let (_, _) = hanging_load(&mut world, 2.0, 2.0);
    let constraint = world.constraints()[0];
    for _ in 0..180 {
        world.step(DT);
    }
    let report = world.inspect_constraint_force(constraint);
    let weight = 2.0 * 9.81;
    let magnitude = (report.force_on_second[0].powi(2)
        + report.force_on_second[1].powi(2)
        + report.force_on_second[2].powi(2))
    .sqrt();
    assert!(
        (magnitude - weight).abs() < weight * 0.02,
        "a joint carrying {weight} N must report it, got {magnitude}"
    );
    assert!(
        report.force_on_second[1] > 0.0,
        "the reaction on the hanging body must pull it up, got {:?}",
        report.force_on_second
    );
    assert!(
        report.torque_on_second.iter().all(|axis| axis.abs() < 0.01),
        "a centric load carries no torque, got {:?}",
        report.torque_on_second
    );
}

#[test]
fn an_unloaded_joint_reports_no_reaction() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]));
    let constraint = world.add_constraint(
        first,
        second,
        ConstraintDesc::ball([0.0; 3], [0.0, -1.0, 0.0]),
    );
    for _ in 0..120 {
        world.step(DT);
    }
    let force = reaction_of(&mut world, constraint);
    assert!(
        force.iter().all(|axis| axis.abs() < 1e-3),
        "an unloaded joint carries no force, got {force:?}"
    );
}

#[test]
fn a_joint_reaction_is_the_same_under_any_substep_budget() {
    for substeps in [1u32, 8] {
        let mut world = new_world(PhysicsConfig {
            substeps,
            ..gravity_config()
        });
        hanging_load(&mut world, 2.0, 2.0);
        let constraint = world.constraints()[0];
        for _ in 0..180 {
            world.step(DT);
        }
        let force = reaction_of(&mut world, constraint);
        let magnitude = force[1];
        assert!(
            (magnitude - 19.62).abs() < 0.4,
            "a joint must report the same {substeps}-substep reaction, got {magnitude}"
        );
    }
}

#[test]
fn a_break_verdict_ignores_the_substep_budget() {
    let weight = 2.0 * 9.81;
    for substeps in [1u32, 8] {
        let mut world = new_world(PhysicsConfig {
            substeps,
            ..gravity_config()
        });
        let anchor = world.spawn(BodyDesc::static_sphere(0.1).position([0.0, 5.0, 0.0]));
        let load = world.spawn(BodyDesc::sphere(0.2).mass(2.0).position([0.0, 3.0, 0.0]));
        let joint = world.add_constraint(
            anchor,
            load,
            ConstraintDesc::distance([0.0; 3], [0.0; 3], 2.0).break_threshold(weight * 0.5, 0.0),
        );
        let mut carrying = new_world(PhysicsConfig {
            substeps,
            ..gravity_config()
        });
        let anchor = carrying.spawn(BodyDesc::static_sphere(0.1).position([0.0, 5.0, 0.0]));
        let held = carrying.spawn(BodyDesc::sphere(0.2).mass(2.0).position([0.0, 3.0, 0.0]));
        carrying.add_constraint(
            anchor,
            held,
            ConstraintDesc::distance([0.0; 3], [0.0; 3], 2.0).break_threshold(weight * 1.5, 0.0),
        );
        for _ in 0..180 {
            world.step(DT);
            carrying.step(DT);
        }
        let broken = world.drain_constraint_breaks();
        assert!(
            broken.contains(&joint),
            "a joint loaded past half its weight limit must break under {substeps} substeps"
        );
        assert!(
            carrying.drain_constraint_breaks().is_empty(),
            "a joint loaded below its limit must survive {substeps} substeps"
        );
    }
}

#[test]
fn a_ball_chain_holds_its_anchors_at_a_hundred_to_one_mass_ratio() {
    let mut world = new_world(gravity_config());
    let anchor = world.spawn(BodyDesc::sphere(0.2).mass(0.0));
    let mut previous = anchor;
    let mut links = Vec::new();
    for index in 0..4 {
        let mass = if index == 3 { 100.0 } else { 1.0 };
        let link = world.spawn(BodyDesc::cuboid([0.1, 0.4, 0.1]).mass(mass).position([
            0.0,
            -0.8 * (index + 1) as f32,
            0.0,
        ]));
        world.add_constraint(
            previous,
            link,
            ConstraintDesc::ball([0.0, -0.4, 0.0], [0.0, 0.4, 0.0]),
        );
        previous = link;
        links.push(link);
    }
    for _ in 0..300 {
        world.step(DT);
    }
    world.wait();
    let mut cursor = anchor;
    let mut worst = 0.0f32;
    for link in &links {
        let above = world.read_state(cursor).position;
        let below = world.read_state(*link).position;
        let distance = ((above[0] - below[0]).powi(2)
            + (above[1] - below[1]).powi(2)
            + (above[2] - below[2]).powi(2))
        .sqrt();
        worst = worst.max((distance - 0.8).abs());
        cursor = *link;
    }
    assert!(
        worst < 0.05,
        "a hundred to one ball chain must keep its anchors together, worst gap {worst}"
    );
}
