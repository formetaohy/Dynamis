use super::common::{DT, settle, sim, static_config};
use dynamis_model::{BodyDesc, QueryFilter, Shape};
use dynamis_simulate::Simulation;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn query_static(
    world: &mut Simulation,
    radius: f32,
    position: [f32; 3],
) -> dynamis_model::BodyHandle {
    world.spawn(BodyDesc::static_sphere(radius).position(position))
}

#[test]
fn ray_hits_nearest_and_reports_surface() {
    let mut world = sim(4, static_config());
    let near = query_static(&mut world, 0.5, [0.0, 0.0, 2.0]);
    let _far = query_static(&mut world, 0.5, [0.0, 0.0, 10.0]);
    let query = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let hit = world.query_hit(query).expect("raycast must hit");
    assert_eq!(hit.body, near);
    assert!((hit.distance - 1.5).abs() < 1e-3);
    assert!((hit.point[2] - 1.5).abs() < 1e-3);
    assert!(
        (hit.normal[2] + 1.0).abs() < 1e-3,
        "normal must face the ray"
    );
    assert_eq!(hit.step, 0);
}

#[test]
fn ray_miss_variants() {
    let mut world = sim(4, static_config());
    let _target = query_static(&mut world, 0.5, [0.0, 0.0, 10.0]);
    let short = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        5.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    assert_eq!(world.query_hit(short), None, "out-of-range ray must miss");

    let mut behind = sim(4, static_config());
    query_static(&mut behind, 0.5, [0.0, 0.0, -5.0]);
    let backward = behind.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    behind.step(DT);
    behind.wait();
    assert_eq!(
        behind.query_hit(backward),
        None,
        "behind-body ray must miss"
    );
}

#[test]
fn ray_from_inside_body_returns_exit_distance() {
    let mut world = sim(4, static_config());
    let target = query_static(&mut world, 0.5, [0.0, 0.0, 0.0]);
    let query = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let hit = world.query_hit(query).expect("origin inside body must hit");
    assert_eq!(hit.body, target);
    assert!(
        (hit.distance - 0.5).abs() < 1e-3,
        "inside ray must exit at the radius"
    );
}

#[test]
fn sphere_query_reports_penetration_and_miss() {
    let mut world = sim(4, static_config());
    let target = query_static(&mut world, 0.5, [0.0, 0.0, 0.0]);
    let overlap = world.sphere_query([0.0, 0.0, 0.8], 0.5, &QueryFilter::default());
    let miss = world.sphere_query([0.0, 0.0, 5.0], 0.5, &QueryFilter::default());
    world.step(DT);
    world.wait();
    let hit = world.query_hit(overlap).expect("overlap must hit");
    assert_eq!(hit.body, target);
    assert!(
        (hit.distance - (-0.2)).abs() < 1e-3,
        "penetration depth must be -0.2, got {}",
        hit.distance
    );
    assert_eq!(world.query_hit(miss), None);
}

#[test]
fn cuboid_query_reports_overlap_and_outside() {
    let mut world = sim(4, static_config());
    query_static(&mut world, 1.0, [0.0, 0.0, 0.0]);
    let inside = world.cuboid_query([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], &QueryFilter::default());
    let outside = world.cuboid_query([0.0, 0.0, 10.0], [1.0, 1.0, 1.0], &QueryFilter::default());
    world.step(DT);
    world.wait();
    assert!(world.query_hit(inside).is_some(), "box overlap must hit");
    assert_eq!(world.query_hit(outside), None, "disjoint box must miss");
}

#[test]
fn sweep_query_stops_at_surface() {
    let mut world = sim(8, static_config());
    let wall = query_static(&mut world, 2.0, [0.0, 0.0, 10.0]);
    let query = world.sweep_query(
        &Shape::sphere(0.3),
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        30.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let hit = world.query_hit(query).expect("sweep must hit the wall");
    assert_eq!(hit.body, wall);
    assert!(
        (hit.distance - 7.7).abs() < 0.05,
        "sweep must stop at the surface (7.7), got {}",
        hit.distance
    );
}

#[test]
fn filters_skip_each_body_kind() {
    let mut world = sim(8, static_config());
    query_static(&mut world, 0.5, [0.0, 0.0, 2.0]);
    let kinematic = world.spawn(
        BodyDesc::sphere(0.5)
            .kinematic(true)
            .position([0.0, 0.0, 4.0]),
    );
    let sensor = world.spawn(BodyDesc::sphere(0.5).sensor(true).position([0.0, 0.0, 6.0]));
    let dynamic = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 8.0]));
    world.sleep(dynamic);
    settle(&mut world, 1);

    let plain = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    let no_static = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            ignore_static: true,
            ..QueryFilter::default()
        },
    );
    let no_static_kinematic = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            ignore_static: true,
            ignore_kinematic: true,
            ..QueryFilter::default()
        },
    );
    let also_no_sleeping = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            ignore_static: true,
            ignore_kinematic: true,
            ignore_sleeping: true,
            ..QueryFilter::default()
        },
    );
    let include_sensors = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            ignore_static: true,
            ignore_kinematic: true,
            ignore_sleeping: true,
            ignore_sensors: false,
            ..QueryFilter::default()
        },
    );
    world.step(DT);
    world.wait();

    let plain = world
        .query_hit(plain)
        .expect("static must be hit by default");
    assert!((plain.distance - 1.5).abs() < 1e-3);
    let no_static = world.query_hit(no_static).expect("kinematic must be hit");
    assert_eq!(no_static.body, kinematic);
    assert!((no_static.distance - 3.5).abs() < 1e-3);
    let no_static_kinematic = world
        .query_hit(no_static_kinematic)
        .expect("dynamic must be hit even while sleeping by default");
    assert_eq!(no_static_kinematic.body, dynamic);
    assert!((no_static_kinematic.distance - 7.5).abs() < 1e-3);
    assert_eq!(
        world.query_hit(also_no_sleeping),
        None,
        "sleeping body must be skipped when ignored"
    );
    let sensor_hit = world
        .query_hit(include_sensors)
        .expect("sensor must be hit when not ignored");
    assert_eq!(sensor_hit.body, sensor);
    assert!((sensor_hit.distance - 5.5).abs() < 1e-3);
}

#[test]
fn group_and_mask_filters_select_bodies() {
    let mut world = sim(4, static_config());
    let a = world.spawn(
        BodyDesc::static_sphere(0.5)
            .position([0.0, 0.0, 2.0])
            .collision_group(1)
            .collision_mask(1),
    );
    let b = world.spawn(
        BodyDesc::static_sphere(0.5)
            .position([0.0, 0.0, 4.0])
            .collision_group(2)
            .collision_mask(2),
    );
    let group_one = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            group: 1,
            mask: 0xFFFF_FFFF,
            ..QueryFilter::default()
        },
    );
    let group_two = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            group: 2,
            mask: 0xFFFF_FFFF,
            ..QueryFilter::default()
        },
    );
    let mask_excludes_a = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            group: 1,
            mask: 2,
            ..QueryFilter::default()
        },
    );
    world.step(DT);
    world.wait();
    let one = world.query_hit(group_one).expect("group 1 must hit");
    assert_eq!(one.body, a);
    let two = world.query_hit(group_two).expect("group 2 must hit");
    assert_eq!(two.body, b);
    let excluded = world.query_hit(mask_excludes_a);
    assert_eq!(
        excluded, None,
        "mask must exclude bodies whose group does not intersect it"
    );
}

#[test]
fn multi_hit_query_reports_all_in_distance_order() {
    let mut world = sim(8, static_config());
    query_static(&mut world, 0.4, [0.0, 0.0, 2.0]);
    query_static(&mut world, 0.4, [0.0, 0.0, 2.7]);
    query_static(&mut world, 0.4, [0.0, 0.0, 3.4]);
    let all = world.sphere_query(
        [0.0, 0.0, 2.7],
        1.2,
        &QueryFilter {
            max_hits: 4,
            ..QueryFilter::default()
        },
    );
    let capped = world.sphere_query(
        [0.0, 0.0, 2.7],
        1.2,
        &QueryFilter {
            max_hits: 1,
            ..QueryFilter::default()
        },
    );
    world.step(DT);
    world.wait();
    let hits = world.query_hits(all);
    assert_eq!(hits.len(), 3, "all three overlaps must be reported");
    assert!(
        hits.windows(2)
            .all(|pair| pair[0].distance <= pair[1].distance)
    );
    assert!(!world.query_overflow(all));
    assert_eq!(world.query_hits(capped).len(), 1);
    assert!(world.query_overflow(capped), "capped query must overflow");
}

#[test]
fn batched_queries_resolve_in_submission_order() {
    let mut world = sim(4, static_config());
    let near = query_static(&mut world, 0.5, [0.0, 0.0, 2.0]);
    let far = query_static(&mut world, 0.5, [0.0, 0.0, 8.0]);
    let first = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        30.0,
        &QueryFilter::default(),
    );
    let second = world.ray_query(
        [0.0, 0.0, 9.5],
        [0.0, 0.0, -1.0],
        30.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    assert_eq!(world.query_hit(first).expect("first must hit").body, near);
    assert_eq!(world.query_hit(second).expect("second must hit").body, far);
}

#[test]
fn results_persist_until_slot_reused() {
    let mut world = sim(2, static_config());
    let target = query_static(&mut world, 0.5, [0.0, 0.0, 2.0]);
    let query = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        10.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let hit = world.query_hit(query).expect("first must hit");
    assert_eq!(hit.body, target);
    settle(&mut world, 5);
    assert_eq!(world.query_hit(query), Some(hit));
}

#[test]
fn retired_batch_invalidates_handle() {
    let mut world = sim(2, static_config());
    let first = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        10.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let _ = world.query_hit(first);
    for _ in 0..4 {
        world.ray_query(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            10.0,
            &QueryFilter::default(),
        );
        world.step(DT);
        world.wait();
    }
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.query_hit(first))).is_err(),
        "a handle whose batch has been retired must stop resolving"
    );
}

#[test]
fn handle_without_arrived_results_panics() {
    let mut world = sim(2, static_config());
    let pending = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        10.0,
        &QueryFilter::default(),
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.query_hit(pending))).is_err(),
        "reading a query before its step has run must panic"
    );
}

#[test]
fn raycast_resolves_against_latest_state() {
    let mut world = sim(4, static_config());
    let target = query_static(&mut world, 0.5, [0.0, 0.0, 2.0]);
    world.step(DT);
    world.wait();
    world.set_position(target, [0.0, 0.0, 30.0]);
    let query = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        10.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    assert_eq!(
        world.query_hit(query),
        None,
        "query must run against the post-teleport state"
    );
}

#[test]
fn query_validation_panics() {
    let mut world = sim(4, static_config());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.sphere_query(
                [0.0, 0.0, 0.0],
                1.0,
                &QueryFilter {
                    max_hits: 32,
                    ..QueryFilter::default()
                },
            );
        }))
        .is_err(),
        "a request above the hit lane count must fail fast"
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.ray_query(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                0.0,
                &QueryFilter::default(),
            );
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.ray_query(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                10.0,
                &QueryFilter::default(),
            );
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.sphere_query([0.0, 0.0, 0.0], 0.0, &QueryFilter::default());
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.cuboid_query([0.0; 3], [0.0, 1.0, 1.0], &QueryFilter::default());
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.sweep_query(
                &Shape::sphere(0.3),
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                0.0,
                &QueryFilter::default(),
            );
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.sweep_query(
                &Shape::sphere(0.3),
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                10.0,
                &QueryFilter::default(),
            );
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.sweep_query(
                &Shape::sphere(0.3),
                [1.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                10.0,
                &QueryFilter::default(),
            );
        }))
        .is_err()
    );
}

#[test]
fn point_query_detects_inside_and_outside() {
    let mut world = sim(6, static_config());
    let body = query_static(&mut world, 0.5, [0.0, 0.0, 0.0]);
    world.step(DT);
    world.wait();
    let inside = world.point_query([0.1, 0.0, 0.0], &QueryFilter::default());
    world.flush_queries();
    let inside_hit = world.query_hit(inside);
    assert_eq!(inside_hit.map(|hit| hit.body), Some(body));
    let outside = world.point_query([5.0, 0.0, 0.0], &QueryFilter::default());
    world.flush_queries();
    assert!(
        world.query_hit(outside).is_none(),
        "point outside must miss"
    );
    let edge = world.point_query([0.5, 0.0, 0.0], &QueryFilter::default());
    world.flush_queries();
    assert!(
        world.query_hit(edge).is_some(),
        "point on the surface must hit"
    );
}

#[test]
fn overlap_query_accepts_any_convex_shape() {
    let mut world = sim(6, static_config());
    let body = query_static(&mut world, 0.5, [0.0, 0.0, 0.0]);
    world.add_collider(
        body,
        dynamis_model::ColliderDesc::new(Shape::cylinder(0.4, 0.3)).offset([2.0, 0.0, 0.0]),
    );
    world.step(DT);
    world.wait();
    let probe = Shape::cylinder(0.2, 0.2);
    let handle = world.overlap_query(
        &probe,
        [0.0, 0.0, 0.0, 1.0],
        [2.2, 0.0, 0.0],
        &QueryFilter::default(),
    );
    world.flush_queries();
    let hit = world.query_hit(handle);
    assert_eq!(hit.map(|hit| (hit.body, hit.collider)), Some((body, 1)));
    let miss = world.overlap_query(
        &probe,
        [0.0, 0.0, 0.0, 1.0],
        [8.0, 0.0, 0.0],
        &QueryFilter::default(),
    );
    world.flush_queries();
    assert!(world.query_hit(miss).is_none());
}

#[test]
fn include_filter_limits_results_to_one_body() {
    let mut world = sim(6, static_config());
    let first = query_static(&mut world, 0.5, [0.0, 0.0, 0.0]);
    let second = query_static(&mut world, 0.5, [3.0, 0.0, 0.0]);
    world.step(DT);
    world.wait();
    let filter = QueryFilter {
        include: Some(second),
        exclude: None,
        max_hits: 4,
        ..Default::default()
    };
    let handle = world.ray_query([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, &filter);
    world.flush_queries();
    let hits = world.query_hits(handle);
    assert!(!hits.is_empty(), "include query must not be empty");
    assert!(
        hits.iter().all(|hit| hit.body == second),
        "include filter must only return the target body"
    );
    assert!(!hits.iter().any(|hit| hit.body == first));
}

#[test]
fn capsule_down_sweep_normal_is_vertical() {
    let mut world = sim(6, static_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    world.step(DT);
    world.wait();
    let probe = Shape::capsule(0.4, 0.5);
    for x in [0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0] {
        let handle = world.sweep_query(
            &probe,
            [0.0, 0.0, 0.0, 1.0],
            [x, 0.95, 0.0],
            [0.0, -1.0, 0.0],
            0.5,
            &QueryFilter::default(),
        );
        world.flush_queries();
        let hit = world
            .query_hit(handle)
            .expect("down sweep must hit the floor");
        eprintln!("x={x} d={} n={:?}", hit.distance, hit.normal);
    }
}

#[test]
fn capsule_sweep_stops_before_wall_face() {
    let mut world = sim(6, static_config());
    world.spawn(
        BodyDesc::cuboid([0.25, 3.0, 4.0])
            .mass(0.0)
            .position([2.0, 1.5, 0.0]),
    );
    world.step(DT);
    world.wait();
    let probe = Shape::sphere(0.4);
    let handle = world.sweep_query(
        &probe,
        [0.0, 0.0, 0.0, 1.0],
        [1.3, 0.95, 0.0],
        [1.0, 0.0, 0.0],
        0.5,
        &QueryFilter::default(),
    );
    world.flush_queries();
    let hit = world
        .query_hit(handle)
        .expect("sweep must stop at the wall");
    assert!(
        hit.distance > 0.0 && hit.distance < 0.25,
        "sweep must stop short of the wall face, got {}",
        hit.distance
    );
    assert!(
        hit.normal[0] > 0.99,
        "wall normal must face the capsule, got {:?}",
        hit.normal
    );
}

#[test]
fn query_after_teleport_sees_the_new_position() {
    let mut world = sim(8, static_config());
    let body = query_static(&mut world, 0.5, [0.0, 0.0, 10.0]);

    world.set_position(body, [0.0, 0.0, 2.0]);
    let query = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    world.flush_queries();
    let hit = world
        .query_hit(query)
        .expect("ray must hit the teleported body");
    assert_eq!(hit.body, body);
    assert!((hit.distance - 1.5).abs() < 1e-3, "got {}", hit.distance);
    assert!(!world.query_overflow(query));

    let stale = world.point_query([0.0, 0.0, 10.0], &QueryFilter::default());
    world.flush_queries();
    assert!(
        world.query_hit(stale).is_none(),
        "stale entries must not answer for the old position"
    );
}
