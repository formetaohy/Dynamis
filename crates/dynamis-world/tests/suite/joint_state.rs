use super::common::{DT, new_world, settle, static_config};
use dynamis_model::{
    BodyDesc, ConstraintDesc, ConstraintKind, ConstraintLimit, ConstraintMotor, DofDesc, JointDof,
};
use dynamis_world::World;
use std::f32::consts::FRAC_PI_6;

fn hinge_world() -> (
    World,
    dynamis_model::BodyHandle,
    dynamis_model::ConstraintHandle,
) {
    let mut world = new_world(static_config());
    let anchor = world.spawn(BodyDesc::sphere(0.1).mass(0.0).position([0.0, 3.0, 0.0]));
    let arm = world.spawn(BodyDesc::sphere(0.2).position([1.0, 3.0, 0.0]));
    let joint = world.add_constraint(
        anchor,
        arm,
        ConstraintDesc::revolute([0.0; 3], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    );
    (world, arm, joint)
}

fn spin_about_z(angle: f32) -> [f32; 4] {
    [0.0, 0.0, (0.5 * angle).sin(), (0.5 * angle).cos()]
}

#[test]
fn revolute_reports_the_hinge_angle() {
    let (mut world, arm, joint) = hinge_world();
    world.set_orientation(arm, spin_about_z(FRAC_PI_6));
    settle(&mut world, 2);
    let state = world.inspect_joint_state(joint);
    assert_eq!(state.kind(), ConstraintKind::Revolute);
    assert_eq!(state.dofs(), &[JointDof::Hinge]);
    assert!(
        (state.coordinate(JointDof::Hinge) - FRAC_PI_6).abs() < 1e-3,
        "the hinge coordinate must report the relative angle, got {}",
        state.coordinate(JointDof::Hinge)
    );
}

#[test]
fn revolute_reports_the_hinge_rate() {
    let (mut world, arm, joint) = hinge_world();
    world.set_angular_velocity(arm, [0.0, 0.0, 1.5]);
    world.set_velocity(arm, [0.0, 1.5, 0.0]);
    world.step(DT);
    world.wait();
    let state = world.inspect_joint_state(joint);
    assert!(
        (state.rate(JointDof::Hinge) - 1.5).abs() < 1e-3,
        "the hinge rate must report the relative angular speed, got {}",
        state.rate(JointDof::Hinge)
    );
}

#[test]
fn servo_tracks_the_reported_hinge_angle() {
    let (mut world, _arm, joint) = hinge_world();
    world.set_motor(joint, 0.0, 100.0);
    world.set_servo(joint, 0.5, 0.5, 0.9);
    settle(&mut world, 120);
    let state = world.inspect_joint_state(joint);
    let angle = state.coordinate(JointDof::Hinge);
    assert!(
        (angle - 0.5).abs() < 0.02,
        "the servo must drive the reported hinge angle, got {angle}"
    );
    assert!(
        state.rate(JointDof::Hinge).abs() < 0.02,
        "a settled servo must report a still hinge, got {}",
        state.rate(JointDof::Hinge)
    );
}

#[test]
fn revolute_limit_reports_the_bounded_coordinate() {
    let (mut world, arm, joint) = hinge_world();
    world.set_limit(
        joint,
        Some(ConstraintLimit {
            min: -0.25,
            max: 0.25,
        }),
    );
    world.set_angular_velocity(arm, [0.0, 0.0, 4.0]);
    world.set_velocity(arm, [0.0, 4.0, 0.0]);
    settle(&mut world, 60);
    let angle = world.inspect_joint_state(joint).coordinate(JointDof::Hinge);
    assert!(
        (0.2..=0.3).contains(&angle),
        "the hinge must report a coordinate inside its limit, got {angle}"
    );
}

#[test]
fn prismatic_reports_the_slide() {
    let mut world = new_world(static_config());
    let base = world.spawn(BodyDesc::sphere(0.1).mass(0.0));
    let slider = world.spawn(
        BodyDesc::sphere(0.2)
            .position([0.0, 0.0, 0.5])
            .velocity([0.0, 0.0, -0.75]),
    );
    let joint = world.add_constraint(
        base,
        slider,
        ConstraintDesc::prismatic([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]),
    );
    world.step(DT);
    world.wait();
    let state = world.inspect_joint_state(joint);
    assert!(
        (state.coordinate(JointDof::Slide) - (0.5 - 0.75 * DT)).abs() < 1e-3,
        "the slide coordinate must report the integrated extension, got {}",
        state.coordinate(JointDof::Slide)
    );
    assert!(
        (state.rate(JointDof::Slide) + 0.75).abs() < 1e-3,
        "the slide rate must report the extension speed, got {}",
        state.rate(JointDof::Slide)
    );
}

#[test]
fn prismatic_servo_reaches_the_reported_slide() {
    let mut world = new_world(static_config());
    let base = world.spawn(BodyDesc::sphere(0.1).mass(0.0));
    let slider = world.spawn(BodyDesc::sphere(0.2).position([0.0, 0.0, 0.5]));
    let joint = world.add_constraint(
        base,
        slider,
        ConstraintDesc::prismatic([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]),
    );
    world.set_motor(joint, 0.0, 50.0);
    world.set_servo(joint, 0.25, 0.5, 0.9);
    settle(&mut world, 120);
    let extension = world.inspect_joint_state(joint).coordinate(JointDof::Slide);
    assert!(
        (extension - 0.25).abs() < 0.01,
        "the slide servo must reach its reported target, got {extension}"
    );
}

#[test]
fn distance_reports_the_separation() {
    let mut world = new_world(static_config());
    let anchor = world.spawn(BodyDesc::sphere(0.1).mass(0.0).position([0.0, 2.0, 0.0]));
    let weight = world.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]));
    let joint = world.add_constraint(
        anchor,
        weight,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0),
    );
    settle(&mut world, 2);
    let state = world.inspect_joint_state(joint);
    assert_eq!(state.dofs(), &[JointDof::Separation]);
    assert!(
        (state.coordinate(JointDof::Separation) - 1.0).abs() < 1e-3,
        "the separation must report the anchor distance, got {}",
        state.coordinate(JointDof::Separation)
    );
}

#[test]
fn six_dof_reports_each_locked_dof() {
    let mut world = new_world(static_config());
    let base = world.spawn(BodyDesc::sphere(0.1).mass(0.0));
    let link = world.spawn(BodyDesc::sphere(0.2).position([0.0, 0.0, 0.3]));
    let joint = world.add_constraint(
        base,
        link,
        ConstraintDesc::six_dof([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]).dofs([
            DofDesc::free(),
            DofDesc::free(),
            DofDesc::limited(0.0, 1.0),
            DofDesc::free(),
            DofDesc::free(),
            DofDesc::free(),
        ]),
    );
    settle(&mut world, 2);
    let state = world.inspect_joint_state(joint);
    assert_eq!(state.dofs().len(), 6);
    assert!(
        (state.coordinate(JointDof::Linear(2)) - 0.3).abs() < 1e-3,
        "the six dof hinge lane must report the axial offset, got {}",
        state.coordinate(JointDof::Linear(2))
    );
    assert!(
        state.coordinate(JointDof::Linear(0)).abs() < 1e-3
            && state.coordinate(JointDof::Angular(0)).abs() < 1e-3,
        "an aligned six dof must report zeroed tangent lanes"
    );
}

#[test]
fn six_dof_drive_reaches_the_reported_dof() {
    let mut world = new_world(static_config());
    let base = world.spawn(BodyDesc::sphere(0.1).mass(0.0));
    let link = world.spawn(BodyDesc::sphere(0.2).position([0.0, 0.0, 0.3]));
    let joint = world.add_constraint(
        base,
        link,
        ConstraintDesc::six_dof([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]).dofs([
            DofDesc::free(),
            DofDesc::free(),
            DofDesc::driven(ConstraintMotor {
                target_velocity: 0.0,
                max_force: 50.0,
                target_position: Some(0.5),
                stiffness: 0.5,
                damping: 0.9,
            }),
            DofDesc::free(),
            DofDesc::free(),
            DofDesc::free(),
        ]),
    );
    settle(&mut world, 120);
    let extension = world
        .inspect_joint_state(joint)
        .coordinate(JointDof::Linear(2));
    assert!(
        (extension - 0.5).abs() < 0.02,
        "the six dof drive must reach its reported lane, got {extension}"
    );
}

#[test]
fn inspect_joint_states_report_every_live_constraint() {
    let (mut world, arm, hinge) = hinge_world();
    let second = world.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]));
    let rope = world.add_constraint(
        arm,
        second,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0),
    );
    settle(&mut world, 2);
    let states = world.inspect_joint_states();
    assert_eq!(states.len(), 2);
    assert_eq!(states[0].0, hinge);
    assert_eq!(states[0].1.kind(), ConstraintKind::Revolute);
    assert_eq!(states[1].0, rope);
    assert_eq!(states[1].1.kind(), ConstraintKind::Distance);
    let _ = rope;
    assert!(
        world
            .inspect_joint_state(hinge)
            .impulse(JointDof::Hinge)
            .is_finite()
    );
}

#[test]
fn a_removed_constraint_has_no_inspect_joint_state() {
    let (mut world, _arm, joint) = hinge_world();
    settle(&mut world, 2);
    world.remove_constraint(joint);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.inspect_joint_state(joint)
    }));
    assert!(result.is_err(), "a stale joint handle must be refused");
}

#[test]
fn pulley_reports_a_held_rope_rate() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.2).position([2.0, 1.0, 0.0]));
    let joint = world.add_constraint(
        first,
        second,
        ConstraintDesc::pulley([0.0; 3], [0.0; 3], [0.0, 3.0, 0.0], [2.0, 3.0, 0.0], 4.0),
    );
    for _ in 0..60 {
        world.apply_force(first, [0.0, 60.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let rate = world.inspect_joint_state(joint).rate(JointDof::Rope);
    assert!(
        rate.abs() < 1e-4,
        "a held pulley must report a still rope, got rate {rate}"
    );
}

#[test]
fn every_joint_kind_reports_its_dof_layout() {
    let mut world = new_world(static_config());
    let descs = [
        ConstraintDesc::ball([0.0; 3], [0.0; 3]),
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0),
        ConstraintDesc::revolute([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]),
        ConstraintDesc::prismatic([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]),
        ConstraintDesc::fixed([0.0; 3], [0.0; 3]),
        ConstraintDesc::gear([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 2.0),
        ConstraintDesc::pulley([0.0; 3], [0.0; 3], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], 1.0),
        ConstraintDesc::cone([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0], 0.5),
        ConstraintDesc::six_dof([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
    ];
    let expected = [
        ConstraintKind::Ball,
        ConstraintKind::Distance,
        ConstraintKind::Revolute,
        ConstraintKind::Prismatic,
        ConstraintKind::Fixed,
        ConstraintKind::Gear,
        ConstraintKind::Pulley,
        ConstraintKind::Cone,
        ConstraintKind::SixDof,
    ];
    for (index, desc) in descs.into_iter().enumerate() {
        let base = world.spawn(BodyDesc::sphere(0.1).mass(0.0));
        let link = world.spawn(BodyDesc::sphere(0.2).position([0.0, 0.0, 0.5]));
        world.add_constraint(base, link, desc);
        assert_eq!(
            world.constraints().len(),
            index + 1,
            "each kind must attach its own constraint"
        );
    }
    settle(&mut world, 2);
    let states = world.inspect_joint_states();
    assert_eq!(states.len(), descs.len());
    for (index, kind) in expected.iter().enumerate() {
        let state = &states[index].1;
        assert_eq!(state.kind(), *kind);
        assert!(
            state.dofs().is_empty() == matches!(kind, ConstraintKind::Fixed | ConstraintKind::Gear),
            "only the fixed and gear joints carry no dof"
        );
        for dof in state.dofs() {
            assert!(
                state.coordinate(*dof).is_finite() && state.rate(*dof).is_finite(),
                "the {kind:?} dof {dof:?} must report finite values"
            );
        }
    }
}

fn two_joint_world() -> (
    World,
    dynamis_model::ConstraintHandle,
    dynamis_model::ConstraintHandle,
) {
    let (mut world, arm, hinge) = hinge_world();
    let link = world.spawn(BodyDesc::sphere(0.2).position([2.0, 3.0, 0.0]));
    let rope = world.add_constraint(arm, link, ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0));
    (world, hinge, rope)
}

fn poll_joint(
    world: &mut World,
    handle: dynamis_model::ConstraintHandle,
) -> dynamis_model::JointState {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        world.poll();
        if let Some(observed) = world.try_joint_state(handle) {
            return observed.value;
        }
        std::thread::yield_now();
    }
    panic!("a watched joint must reach the host without a sync");
}

#[test]
fn a_single_joint_subscription_publishes_only_that_joint() {
    let (mut world, hinge, _rope) = two_joint_world();
    world.try_joint_state(hinge);
    settle(&mut world, 4);
    assert_eq!(
        poll_joint(&mut world, hinge).kind(),
        ConstraintKind::Revolute
    );
    assert_eq!(
        world
            .try_joint_states()
            .expect("a single joint subscription must publish")
            .value
            .len(),
        1,
        "watching one joint must not publish the rest of the constraint set"
    );
    settle(&mut world, 4);
    assert_eq!(
        world
            .try_joint_states()
            .expect("a widened subscription must publish")
            .value
            .len(),
        2,
        "a broad subscription must publish every live joint"
    );
}

#[test]
fn a_single_joint_inspection_publishes_only_the_joint_it_names() {
    let (mut world, hinge, _rope) = two_joint_world();
    settle(&mut world, 2);
    let state = world.inspect_joint_state(hinge);
    assert_eq!(state.kind(), ConstraintKind::Revolute);
    let observed = world
        .try_joint_states()
        .expect("an inspection must leave a publication behind");
    assert_eq!(observed.value.len(), 1);
    assert_eq!(observed.value[0].0, hinge);
}

#[test]
fn a_watched_joint_keeps_its_identity_across_row_churn() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.2).position([0.0, 0.0, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]));
    let third = world.spawn(BodyDesc::sphere(0.2).position([0.0, 2.0, 0.0]));
    let doomed = world.add_constraint(
        first,
        second,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0),
    );
    let watched = world.add_constraint(
        second,
        third,
        ConstraintDesc::revolute([0.0; 3], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    );
    world.try_joint_state(watched);
    settle(&mut world, 4);
    assert_eq!(
        poll_joint(&mut world, watched).kind(),
        ConstraintKind::Revolute
    );
    world.remove_constraint(doomed);
    settle(&mut world, 4);
    let observed = poll_joint(&mut world, watched);
    assert_eq!(
        observed.kind(),
        ConstraintKind::Revolute,
        "a watched joint must keep answering after its row moves"
    );
    assert_eq!(observed, world.inspect_joint_state(watched));
}
