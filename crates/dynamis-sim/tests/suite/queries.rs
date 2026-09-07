use super::common::{DT, settle, sim, static_config};
use dynamis_model::{BodyDesc, QueryFilter, Shape};
use dynamis_sim::Simulation;
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
    let query = world.raycast(
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
    let short = world.raycast(
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
    let backward = behind.raycast(
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
    let query = world.raycast(
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
fn box_query_reports_overlap_and_outside() {
    let mut world = sim(4, static_config());
    query_static(&mut world, 1.0, [0.0, 0.0, 0.0]);
    let inside = world.box_query([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], &QueryFilter::default());
    let outside = world.box_query([0.0, 0.0, 10.0], [1.0, 1.0, 1.0], &QueryFilter::default());
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

    let plain = world.raycast(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    let no_static = world.raycast(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            ignore_static: true,
            ..QueryFilter::default()
        },
    );
    let no_static_kinematic = world.raycast(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            ignore_static: true,
            ignore_kinematic: true,
            ..QueryFilter::default()
        },
    );
    let also_no_sleeping = world.raycast(
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
    let include_sensors = world.raycast(
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
    let group_one = world.raycast(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            group: 1,
            mask: 0xFFFF_FFFF,
            ..QueryFilter::default()
        },
    );
    let group_two = world.raycast(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            group: 2,
            mask: 0xFFFF_FFFF,
            ..QueryFilter::default()
        },
    );
    let mask_excludes_a = world.raycast(
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
    let first = world.raycast(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        30.0,
        &QueryFilter::default(),
    );
    let second = world.raycast(
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
    let query = world.raycast(
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
fn ring_reuse_invalidates_stale_handle() {
    let mut world = sim(2, static_config());
    query_static(&mut world, 0.5, [0.0, 0.0, 2.0]);
    let first = world.raycast(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        10.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let _ = world.query_hit(first);
    for _ in 0..world.capacity() * 2 {
        world.raycast(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            10.0,
            &QueryFilter::default(),
        );
    }
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.query_hit(first))).is_err(),
        "recycled slot must invalidate the old handle"
    );
}

#[test]
fn ring_exhaustion_panics_without_results() {
    let mut world = sim(2, static_config());
    for _ in 0..world.capacity() * 2 {
        world.raycast(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            10.0,
            &QueryFilter::default(),
        );
    }
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.raycast(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                10.0,
                &QueryFilter::default(),
            );
        }))
        .is_err(),
        "submitting beyond the query ring must panic"
    );
}

#[test]
fn raycast_resolves_against_latest_state() {
    let mut world = sim(4, static_config());
    let target = query_static(&mut world, 0.5, [0.0, 0.0, 2.0]);
    world.step(DT);
    world.wait();
    world.set_position(target, [0.0, 0.0, 30.0]);
    let query = world.raycast(
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
            world.raycast(
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
            world.raycast(
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
            world.box_query([0.0; 3], [0.0, 1.0, 1.0], &QueryFilter::default());
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
