use super::common::{DT, gravity_config, new_world, settle, static_config};
use dynamis_model::{BodyDesc, BodyHandle, QueryFilter};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn falling_sphere(world: &mut dynamis_world::World) -> BodyHandle {
    world.spawn(BodyDesc::sphere(0.5).position([0.0, 5.0, 0.0]))
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
    assert!(!world.contact_manifolds().is_empty());
    let base = world.submissions();
    let manifolds = world.contact_manifolds();
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
    assert!(
        !world.query_ready(query),
        "a query reports ready only after its results arrive"
    );
    world.resolve_queries();
    assert_eq!(
        world.submissions() - base,
        1,
        "pending queries resolve in one submission"
    );
    world.wait_query(query);
    assert!(
        world.query_ready(query),
        "a resolved query must become ready"
    );
    assert!(
        world.query_hit(query).is_some(),
        "a resolved query must hit"
    );
}
