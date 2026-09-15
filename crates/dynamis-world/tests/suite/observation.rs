use super::common::{DT, gravity_config, new_world, settle, static_config};
use dynamis_model::{BodyDesc, BodyHandle, BodyState, ConstraintDesc, ConstraintKind, QueryFilter};
use dynamis_world::QueryState;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn falling_sphere(world: &mut dynamis_world::World) -> BodyHandle {
    world.spawn(BodyDesc::sphere(0.5).position([0.0, 5.0, 0.0]))
}

fn crowd(world: &mut dynamis_world::World, count: usize) -> Vec<BodyHandle> {
    (0..count)
        .map(|index| {
            world.spawn(BodyDesc::sphere(0.25).position([
                (index % 16) as f32 * 0.6,
                5.0 + (index / 16) as f32 * 0.6,
                0.0,
            ]))
        })
        .collect()
}

fn poll_observation(world: &mut dynamis_world::World, body: BodyHandle) -> BodyState {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        world.poll();
        if let Some(state) = world.try_state(body) {
            return state;
        }
        std::thread::yield_now();
    }
    panic!("an observed body must reach the mirror without a sync");
}

#[test]
fn an_observation_reports_only_the_bodies_it_observes() {
    let mut world = new_world(gravity_config());
    let bodies = crowd(&mut world, 192);
    let observed = bodies[0];
    let untouched = bodies[1];
    world.try_state(observed);
    world.step(DT);
    let state = poll_observation(&mut world, observed);
    assert!(
        state.position[1] < 5.0,
        "an observed body must report its stepped state"
    );
    assert_eq!(
        world.read_state(observed),
        state,
        "an observed body must answer a strict read on its own"
    );
    assert!(
        world.stream_capacity().state.observed < world.count() as u32,
        "the observation streams must follow the observed set, not the world"
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.read_state(untouched))).is_err(),
        "an unobserved body must not claim a current state"
    );
}

#[test]
fn observing_more_bodies_widens_the_observation_streams() {
    let mut world = new_world(gravity_config());
    let bodies = crowd(&mut world, 192);
    for body in &bodies {
        world.try_state(*body);
    }
    world.step(DT);
    let observed = world.stream_capacity().state.observed;
    assert!(
        observed >= bodies.len() as u32,
        "the observation streams must hold every observed body, got {observed}"
    );
    for body in &bodies {
        poll_observation(&mut world, *body);
    }
}

#[test]
fn a_state_mirror_rides_the_step_submission() {
    let mut world = new_world(gravity_config());
    let body = falling_sphere(&mut world);
    settle(&mut world, 40);
    let base = world.submissions();
    for _ in 0..10 {
        world.try_state(body);
        world.step(DT);
        world.poll();
    }
    assert_eq!(
        world.submissions() - base,
        10,
        "an observed state must ride the step submission instead of opening its own"
    );
    assert!(
        world.try_state(body).is_some(),
        "polling must retire observed states"
    );
}

#[test]
fn polling_only_retires_what_the_engine_already_submitted() {
    let mut world = new_world(gravity_config());
    let body = falling_sphere(&mut world);
    settle(&mut world, 10);
    world.try_state(body);
    world.step(DT);
    let base = world.submissions();
    for _ in 0..8 {
        world.poll();
    }
    assert_eq!(
        world.submissions(),
        base,
        "polling must never open a submission of its own"
    );
}

#[test]
fn observed_state_arrives_without_a_sync_while_strict_reads_require_one() {
    let mut world = new_world(gravity_config());
    let body = falling_sphere(&mut world);
    world.wait();
    assert_eq!(world.read_state(body).position, [0.0, 5.0, 0.0]);
    let base = world.submissions();
    world.try_state(body);
    world.step(DT);
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.read_state(body))).is_err(),
        "a strict read must refuse a state that predates the last step"
    );
    let mut observed = None;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        world.poll();
        if let Some(state) = world.try_state(body) {
            observed = Some(state);
            break;
        }
        std::thread::yield_now();
    }
    let observed = observed.expect("a polled mirror must deliver the stepped state");
    assert!(observed.position[1] < 5.0);
    assert_eq!(
        world.submissions() - base,
        1,
        "observed state must arrive on the step that produced it"
    );
    world.wait();
    assert_eq!(world.read_state(body), observed);
}

#[test]
fn a_sync_opens_at_most_one_submission_and_reads_stay_free() {
    let mut world = new_world(gravity_config());
    let bodies = (0..8)
        .map(|index| world.spawn(BodyDesc::sphere(0.5).position([index as f32, 5.0, 0.0])))
        .collect::<Vec<_>>();
    for _ in 0..10 {
        world.step(DT);
    }
    let base = world.submissions();
    world.wait();
    assert!(
        world.submissions() - base <= 1,
        "a sync must not fan out into one submission per stream"
    );
    let base = world.submissions();
    for body in &bodies {
        std::hint::black_box(world.read_state(*body));
    }
    assert_eq!(
        world.submissions(),
        base,
        "a strict read on a synced mirror must not submit"
    );
}

#[test]
fn contact_inspection_costs_a_single_submission() {
    let mut world = new_world(gravity_config());
    world.spawn(BodyDesc::static_sphere(1.0));
    let body = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.4, 0.0]));
    settle(&mut world, 40);
    assert!(!world.inspect_contacts().is_empty());
    let base = world.submissions();
    let manifolds = world.inspect_contacts();
    assert_eq!(
        world.submissions() - base,
        1,
        "an inspection must stage every region in one submission"
    );
    assert!(
        manifolds
            .iter()
            .any(|manifold| manifold.first == body || manifold.second == body)
    );
}

#[test]
fn a_query_rides_the_step_submission() {
    let mut world = new_world(static_config());
    let ground = world.spawn(BodyDesc::static_sphere(1.0));
    world.try_state(ground);
    world.step(DT);
    world.wait();
    let base = world.submissions();
    let query = world.ray_query(
        [0.0, 5.0, 0.0],
        [0.0, -1.0, 0.0],
        20.0,
        &QueryFilter::default(),
    );
    world.try_state(ground);
    world.step(DT);
    world.wait();
    assert_eq!(
        world.submissions() - base,
        1,
        "a query must ride the step submission instead of opening its own"
    );
    assert!(
        world.query_hit(query).is_some(),
        "a query resolved by the step must deliver its hits"
    );
}

#[test]
fn an_observation_outlives_the_batch_that_follows_it() {
    let mut world = new_world(static_config());
    let target = world.spawn(BodyDesc::static_sphere(1.0).position([0.0, 0.0, 2.0]));
    let first = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    world.resolve_queries();
    let second = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    world.resolve_queries();
    world.wait_query(first);
    assert_eq!(
        world.query_state(first),
        QueryState::Retired,
        "an observation must stay readable after a later batch retires"
    );
    assert_eq!(
        world.query_hit(first).map(|hit| hit.body()),
        Some(target),
        "a retained observation must keep its hits"
    );
    world.wait_query(second);
    assert_eq!(world.query_state(second), QueryState::Retired);
}

#[test]
fn an_observation_beyond_the_retention_window_lapses() {
    let mut world = new_world(static_config());
    world.spawn(BodyDesc::static_sphere(1.0).position([0.0, 0.0, 2.0]));
    let mut handles = Vec::new();
    for _ in 0..dynamis_gpu::FACT_LAG + 2 {
        let handle = world.ray_query(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            20.0,
            &QueryFilter::default(),
        );
        world.resolve_queries();
        world.wait_query(handle);
        handles.push(handle);
    }
    let oldest = handles[0];
    assert_eq!(
        world.query_state(oldest),
        QueryState::Lapsed,
        "an observation past the retention window must report a lapse"
    );
    let newest = *handles.last().expect("a batch was issued");
    assert_eq!(world.query_state(newest), QueryState::Retired);
}

#[test]
fn resolving_queries_opens_a_single_submission() {
    let mut world = new_world(static_config());
    world.spawn(BodyDesc::static_sphere(1.0).position([0.0, 0.0, 0.0]));
    let base = world.submissions();
    let query = world.ray_query(
        [0.0, 5.0, 0.0],
        [0.0, -1.0, 0.0],
        20.0,
        &QueryFilter::default(),
    );
    assert_eq!(
        world.query_state(query),
        QueryState::Pending,
        "a query is pending until its results retire"
    );
    world.resolve_queries();
    assert_eq!(
        world.submissions() - base,
        1,
        "pending queries resolve in one submission"
    );
    world.wait();
    assert_eq!(
        world.query_state(query),
        QueryState::Retired,
        "a resolved query must retire"
    );
    assert!(
        world.query_hit(query).is_some(),
        "a resolved query must hit"
    );
}

fn hinge(world: &mut dynamis_world::World) -> (BodyHandle, dynamis_model::ConstraintHandle) {
    let anchor = world.spawn(BodyDesc::static_sphere(0.1).position([0.0, 3.0, 0.0]));
    let arm = world.spawn(BodyDesc::sphere(0.2).position([1.0, 3.0, 0.0]));
    let joint = world.add_constraint(
        anchor,
        arm,
        dynamis_model::ConstraintDesc::revolute([0.0; 3], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    );
    (arm, joint)
}

fn cloth(world: &mut dynamis_world::World) -> dynamis_model::SoftBodyHandle {
    world.spawn(
        BodyDesc::cuboid([4.0, 0.5, 4.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let mut desc =
        dynamis_model::SoftBodyDesc::cloth([4, 4], 0.25, dynamis_model::SoftMaterial::rigid());
    desc.radius = 0.05;
    desc.position = [0.0, 1.0, 0.0];
    world.add_soft_body(desc)
}

fn publish(world: &mut dynamis_world::World, frames: usize) {
    for _ in 0..frames {
        world.step(DT);
        world.poll();
    }
}

fn poll_joint_mirror(
    world: &mut dynamis_world::World,
) -> dynamis_world::Observation<Vec<(dynamis_model::ConstraintHandle, dynamis_model::JointState)>> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        world.poll();
        if let Some(observed) = world.try_joint_states() {
            return observed;
        }
        std::thread::yield_now();
    }
    panic!("a subscribed joint mirror must reach the host without a sync");
}

fn poll_soft_mirror(
    world: &mut dynamis_world::World,
    handle: dynamis_model::SoftBodyHandle,
) -> dynamis_world::Observation<Vec<[f32; 3]>> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        world.poll();
        if let Some(observed) = world.try_soft_particles(handle) {
            return observed;
        }
        std::thread::yield_now();
    }
    panic!("a subscribed soft mirror must reach the host without a sync");
}

#[test]
fn a_joint_mirror_rides_the_step_submission() {
    let mut world = new_world(gravity_config());
    let (arm, joint) = hinge(&mut world);
    settle(&mut world, 8);
    assert!(
        world.try_joint_states().is_none(),
        "a mirror must report nothing before a step publishes it"
    );
    world.try_state(arm);
    publish(&mut world, 1);
    let base = world.submissions();
    publish(&mut world, 8);
    assert_eq!(
        world.submissions() - base,
        8,
        "a joint mirror must ride the step submission instead of opening its own"
    );
    world.wait();
    let observed = world
        .try_joint_states()
        .expect("a subscribed joint mirror must publish every step");
    assert_eq!(
        observed.value,
        world.inspect_joint_states(),
        "a drained mirror must report the same joints as an inspection"
    );
    assert_eq!(
        observed.step,
        world
            .try_state(arm)
            .expect("the body mirror must publish alongside the joint mirror")
            .step,
        "every mirror must report the step it was published at"
    );
    let force = world
        .try_constraint_force(joint)
        .expect("a joint mirror must publish its reaction");
    assert_eq!(force.step, observed.step);
    assert_eq!(force.value, world.inspect_constraint_force(joint));
    assert_eq!(
        world.try_constraint_forces().expect("silhouette").value,
        world.inspect_constraint_forces(),
        "a force mirror must report every joint reaction"
    );
}

#[test]
fn a_joint_mirror_stops_and_resumes_with_its_subscription() {
    let mut world = new_world(gravity_config());
    hinge(&mut world);
    world.try_joint_states();
    publish(&mut world, 4);
    assert!(poll_joint_mirror(&mut world).value.len() == 1);
    world.stop_observing_joints();
    assert!(
        world.try_joint_states().is_none(),
        "a stopped mirror must forget the publication it held"
    );
    publish(&mut world, 1);
    assert_eq!(
        poll_joint_mirror(&mut world).value.len(),
        1,
        "a re-subscribed mirror must publish on the next step"
    );
}

#[test]
fn a_soft_mirror_rides_the_step_submission() {
    let mut world = new_world(gravity_config());
    let soft = cloth(&mut world);
    settle(&mut world, 8);
    assert!(
        world.try_soft_particles(soft).is_none(),
        "a soft mirror must report nothing before a step publishes it"
    );
    publish(&mut world, 1);
    let base = world.submissions();
    publish(&mut world, 8);
    assert_eq!(
        world.submissions() - base,
        8,
        "a soft mirror must ride the step submission instead of opening its own"
    );
    world.wait();
    let particles = world
        .try_soft_particles(soft)
        .expect("a watched soft body must publish every step");
    assert_eq!(
        particles.value,
        world.inspect_soft_particles(soft),
        "a drained mirror must report the same particles as an inspection"
    );
    let elements = world
        .try_soft_elements(soft)
        .expect("a soft mirror must publish its elements");
    assert_eq!(elements.step, particles.step);
    assert_eq!(elements.value, world.inspect_soft_elements(soft));
}

#[test]
fn a_soft_mirror_reports_only_the_soft_bodies_it_observes() {
    let mut world = new_world(gravity_config());
    let first = cloth(&mut world);
    let second = cloth(&mut world);
    world.try_soft_particles(first);
    publish(&mut world, 4);
    assert!(!poll_soft_mirror(&mut world, first).value.is_empty());
    assert!(
        world.try_soft_particles(second).is_none(),
        "an unobserved soft body must not be published"
    );
    publish(&mut world, 1);
    assert!(!poll_soft_mirror(&mut world, second).value.is_empty());
    world.stop_observing_soft_body(first);
    assert!(
        world.try_soft_particles(first).is_none(),
        "a stopped soft observation must forget the particles it held"
    );
}

#[test]
fn a_stopped_body_observation_forgets_its_state() {
    let mut world = new_world(gravity_config());
    let body = falling_sphere(&mut world);
    world.try_state(body);
    settle(&mut world, 4);
    let observed = poll_observation(&mut world, body);
    world.stop_observing_body(body);
    assert!(
        world.try_state(body).is_none(),
        "a stopped body observation must forget the state it held"
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.read_state(body))).is_err(),
        "a stopped body observation must not claim a current state"
    );
    world.try_state(body);
    settle(&mut world, 1);
    let resumed = poll_observation(&mut world, body);
    assert!(
        resumed.position[1] < observed.position[1],
        "a re-observed body must resume its mirror on the next step, got {} after {}",
        resumed.position[1],
        observed.position[1]
    );
}

#[test]
fn a_strict_inspection_retires_its_own_publication() {
    let mut world = new_world(static_config());
    let anchor = world.spawn(BodyDesc::sphere(0.1).mass(0.0).position([0.0, 3.0, 0.0]));
    let arm = world.spawn(BodyDesc::sphere(0.2).position([1.0, 3.0, 0.0]));
    let joint = world.add_constraint(
        anchor,
        arm,
        ConstraintDesc::revolute([0.0; 3], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    );
    world.step(DT);
    let state = world.inspect_joint_state(joint);
    assert_eq!(state.kind(), ConstraintKind::Revolute);
    world.step(DT);
    let states = world.inspect_joint_states();
    assert_eq!(
        states.len(),
        1,
        "an inspection publishes every live joint once"
    );
    assert_eq!(states[0].0, joint);
}

#[test]
fn a_repeated_sync_wait_publishes_nothing_new() {
    let mut world = new_world(gravity_config());
    let _body = falling_sphere(&mut world);
    settle(&mut world, 4);
    let submissions = world.submissions();
    world.wait();
    assert_eq!(
        world.submissions(),
        submissions,
        "a wait over covered bodies must not submit device work"
    );
}

#[test]
fn a_sync_wait_releases_the_bodies_it_published() {
    let mut world = new_world(gravity_config());
    let bodies = crowd(&mut world, 8);
    settle(&mut world, 4);
    let synced = world.read_state(bodies[0]);
    let observed = bodies[1];
    world.try_state(observed);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mirrored = loop {
        world.step(DT);
        world.poll();
        if let Some(state) = world.try_state(observed)
            && state.step > synced.step
        {
            break state;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "an observed body must keep publishing after a sync"
        );
        std::thread::yield_now();
    };
    assert!(mirrored.position[1] < synced.position[1]);
    assert_eq!(
        world
            .try_state(bodies[0])
            .expect("a synced body keeps its mirror")
            .step,
        synced.step,
        "a sync must not leave its bodies observed"
    );
}
