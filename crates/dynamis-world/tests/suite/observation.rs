use super::common::{DT, gravity_config, new_world, settle, static_config};
use dynamis_model::{BodyDesc, BodyHandle, BodyState, QueryFilter};
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
        world.query_hit(first).map(|hit| hit.body),
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
