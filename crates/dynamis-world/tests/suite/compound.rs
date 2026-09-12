use super::common::{DT, new_world, settle, static_config};
use dynamis_model::{BodyDesc, ColliderDesc, QueryFilter, Shape};

const NESTED_COLLIDERS: usize = 64;

#[test]
fn compound_support_many_colliders() {
    let mut world = new_world(super::common::gravity_config());
    let mut desc = BodyDesc::cuboid([0.4, 0.4, 0.4]);
    for index in 0..NESTED_COLLIDERS {
        let lattice = |step: usize| (index / step % 4) as f32 - 1.5;
        desc = desc.collider(ColliderDesc::new(Shape::sphere(0.05)).offset([
            lattice(1) * 0.2,
            lattice(4) * 0.2,
            lattice(16) * 0.2,
        ]));
    }
    assert_eq!(desc.colliders.len(), NESTED_COLLIDERS + 1);
    let body = world.spawn(desc.position([0.0, 10.0, 0.0]));
    let ground = world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    settle(&mut world, 90);
    let rest = world.read_state(body).position[1];
    assert!(
        (rest - 0.4).abs() < 0.1,
        "compound body must rest on its core cuboid, got {rest}"
    );
    assert_eq!(world.read_state(ground).position[1], -0.5);
}

#[test]
fn runtime_added_collider_supports_falling_body() {
    let mut world = new_world(super::common::gravity_config());
    let base = world.spawn(BodyDesc::sphere(0.3).position([0.0, 0.0, 0.0]).mass(0.0));
    world.add_collider(
        base,
        ColliderDesc::new(Shape::sphere(0.3)).offset([2.0, 0.0, 0.0]),
    );
    let ball = world.spawn(BodyDesc::sphere(0.2).position([2.0, 3.0, 0.0]));
    settle(&mut world, 60);
    let rest = world.read_state(ball).position[1];
    assert!(
        (rest - 0.5).abs() < 0.1,
        "runtime-added collider must catch the ball, got {rest}"
    );
}

#[test]
fn add_collider_then_query_detects_it() {
    let mut world = new_world(static_config());
    let body = world.spawn(BodyDesc::sphere(0.3).position([0.0, 0.0, 0.0]));
    world.add_collider(
        body,
        ColliderDesc::new(Shape::sphere(0.3)).offset([2.0, 0.0, 0.0]),
    );
    world.step(DT);
    world.wait();
    let filter = QueryFilter {
        max_hits: 4,
        ..Default::default()
    };
    let handle = world.overlap_query(
        &Shape::sphere(0.2),
        [0.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0],
        &filter,
    );
    world.flush_queries();
    let hit = world.query_hit(handle);
    assert_eq!(
        hit.map(|hit| (hit.body, hit.collider)),
        Some((body, 1)),
        "runtime-added collider must be queryable at its slot"
    );
}

#[test]
fn removed_collider_stops_colliding() {
    let mut world = new_world(static_config());
    let body = world.spawn(BodyDesc::sphere(0.3).position([0.0, 0.0, 0.0]));
    world.add_collider(
        body,
        ColliderDesc::new(Shape::sphere(0.3)).offset([2.0, 0.0, 0.0]),
    );
    world.remove_collider(body, 1);
    world.step(DT);
    world.wait();
    let filter = QueryFilter {
        max_hits: 4,
        ..Default::default()
    };
    let handle = world.overlap_query(
        &Shape::sphere(0.2),
        [0.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0],
        &filter,
    );
    world.flush_queries();
    assert!(
        world.query_hit(handle).is_none(),
        "removed collider must not be queryable"
    );
}

#[test]
fn removed_body_leaves_no_ghost_collider() {
    let mut world = new_world(static_config());
    let ghost = world.spawn(BodyDesc::static_sphere(0.75).position([0.0, 0.0, 0.0]));
    let filter = QueryFilter::default();
    let probe = world.sphere_query([0.0; 3], 0.5, &filter);
    world.flush_queries();
    assert_eq!(
        world.query_hits(probe).len(),
        1,
        "a live collider must answer a probe"
    );

    world.remove(ghost);
    let probe = world.sphere_query([0.0; 3], 0.5, &filter);
    world.flush_queries();
    assert!(
        world.query_hits(probe).is_empty(),
        "a removed body must leave no ghost collider in the pool"
    );
}

#[test]
fn remove_last_collider_panics() {
    let mut world = new_world(static_config());
    let body = world.spawn(BodyDesc::sphere(0.3));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.remove_collider(body, 0);
        }))
        .is_err(),
        "removing the last collider must panic"
    );
}
