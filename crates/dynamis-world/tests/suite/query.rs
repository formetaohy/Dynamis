use super::common::{
    DT, gravity_config, new_world, settle, settle_until, static_config, static_sphere_ground,
};
use dynamis_model::{BodyDesc, ColliderDesc, CollisionFilter, QueryFilter, Shape};
use dynamis_world::{QueryHit, QueryState, World};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn query_static(world: &mut World, radius: f32, position: [f32; 3]) -> dynamis_model::BodyHandle {
    world.spawn(BodyDesc::static_sphere(radius).position(position))
}

#[test]
fn ray_hits_nearest_and_reports_surface() {
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
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

    let mut behind = new_world(static_config());
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
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
    query_static(&mut world, 1.0, [0.0, 0.0, 0.0]);
    let inside = world.cuboid_query([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], &QueryFilter::default());
    let outside = world.cuboid_query([0.0, 0.0, 10.0], [1.0, 1.0, 1.0], &QueryFilter::default());
    world.step(DT);
    world.wait();
    assert!(world.query_hit(inside).is_some(), "box overlap must hit");
    assert_eq!(world.query_hit(outside), None, "disjoint box must miss");
}

fn down_ray(world: &mut World, kind: &str, start: [f32; 3]) -> QueryHit {
    let handle = world.ray_query(start, [0.0, -1.0, 0.0], 20.0, &QueryFilter::default());
    world.step(DT);
    world.wait();
    world
        .query_hit(handle)
        .unwrap_or_else(|| panic!("a {kind} down ray must hit"))
}

#[test]
fn sweep_query_stops_at_surface() {
    let mut world = new_world(static_config());
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
    assert!(
        (hit.normal[2] + 1.0).abs() < 1e-3,
        "sweep normal must face the sweeping shape, got {:?}",
        hit.normal
    );
}

fn down_sweep(
    world: &mut World,
    kind: &str,
    probe: &Shape,
    start: [f32; 3],
    length: f32,
) -> QueryHit {
    let handle = world.sweep_query(
        probe,
        [0.0, 0.0, 0.0, 1.0],
        start,
        [0.0, -1.0, 0.0],
        length,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    world
        .query_hit(handle)
        .unwrap_or_else(|| panic!("a {kind} down sweep must hit"))
}

#[test]
fn sweep_normals_and_depths_agree_across_every_floor_kind() {
    let mut boxed = new_world(static_config());
    boxed.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let boxed_hit = down_sweep(
        &mut boxed,
        "cuboid",
        &Shape::sphere(0.5),
        [0.0, 3.0, 0.0],
        20.0,
    );

    let mut spherical = new_world(static_config());
    spherical.spawn(BodyDesc::static_sphere(2.5).position([0.0, -2.5, 0.0]));
    let sphere_hit = down_sweep(
        &mut spherical,
        "sphere",
        &Shape::sphere(0.5),
        [0.0, 3.0, 0.0],
        20.0,
    );

    let mut planed = new_world(static_config());
    planed.spawn(BodyDesc::new(ColliderDesc::new(Shape::plane())).mass(0.0));
    let plane_hit = down_sweep(
        &mut planed,
        "plane",
        &Shape::sphere(0.5),
        [0.0, 3.0, 0.0],
        20.0,
    );

    let mut meshed = new_world(static_config());
    let quad = meshed.add_mesh(
        &[
            [-5.0, 0.0, -5.0],
            [5.0, 0.0, -5.0],
            [5.0, 0.0, 5.0],
            [-5.0, 0.0, 5.0],
        ],
        &[[0, 1, 2], [0, 2, 3]],
        None,
    );
    meshed.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(quad))).mass(0.0));
    let mesh_hit = down_sweep(
        &mut meshed,
        "mesh",
        &Shape::sphere(0.5),
        [0.0, 3.0, 0.0],
        20.0,
    );

    for (kind, hit) in [
        ("cuboid", boxed_hit),
        ("sphere", sphere_hit),
        ("plane", plane_hit),
        ("mesh", mesh_hit),
    ] {
        assert!(
            (hit.normal[1] - 1.0).abs() < 1e-3,
            "a {kind} floor must report a floor normal facing the sweep, got {:?}",
            hit.normal
        );
        assert!(
            (hit.distance - 2.5).abs() < 1e-2,
            "a {kind} floor must stop the sweep at the surface (2.5), got {}",
            hit.distance
        );
    }
}

#[test]
fn ray_normals_and_depths_agree_across_every_floor_kind() {
    let mut boxed = new_world(static_config());
    boxed.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let boxed_hit = down_ray(&mut boxed, "cuboid", [0.0, 3.0, 0.0]);

    let mut spherical = new_world(static_config());
    spherical.spawn(BodyDesc::static_sphere(2.5).position([0.0, -2.5, 0.0]));
    let sphere_hit = down_ray(&mut spherical, "sphere", [0.0, 3.0, 0.0]);

    let mut planed = new_world(static_config());
    planed.spawn(BodyDesc::new(ColliderDesc::new(Shape::plane())).mass(0.0));
    let plane_hit = down_ray(&mut planed, "plane", [0.0, 3.0, 0.0]);

    let mut meshed = new_world(static_config());
    let quad = meshed.add_mesh(
        &[
            [-5.0, 0.0, -5.0],
            [5.0, 0.0, -5.0],
            [5.0, 0.0, 5.0],
            [-5.0, 0.0, 5.0],
        ],
        &[[0, 1, 2], [0, 2, 3]],
        None,
    );
    meshed.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(quad))).mass(0.0));
    let mesh_hit = down_ray(&mut meshed, "mesh", [0.0, 3.0, 0.0]);

    let mut hulled = new_world(static_config());
    let hull = hulled.add_hull(&slab_vertices(), &slab_triangles());
    hulled.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::hull(hull)))
            .mass(0.0)
            .position([0.0, 0.0, 0.0]),
    );
    let hull_hit = down_ray(&mut hulled, "hull", [0.0, 3.0, 0.0]);

    for (kind, hit) in [
        ("cuboid", boxed_hit),
        ("sphere", sphere_hit),
        ("plane", plane_hit),
        ("mesh", mesh_hit),
        ("hull", hull_hit),
    ] {
        assert!(
            (hit.normal[1] - 1.0).abs() < 1e-3,
            "a {kind} floor must report a floor normal opposing the ray, got {:?}",
            hit.normal
        );
        assert!(
            (hit.distance - 3.0).abs() < 1e-2,
            "a {kind} floor must stop the ray at the surface (3.0), got {}",
            hit.distance
        );
    }
}

fn slab_vertices() -> Vec<[f32; 3]> {
    vec![
        [-5.0, 0.0, -5.0],
        [5.0, 0.0, -5.0],
        [5.0, 0.0, 5.0],
        [-5.0, 0.0, 5.0],
        [-5.0, -1.0, -5.0],
        [5.0, -1.0, -5.0],
        [5.0, -1.0, 5.0],
        [-5.0, -1.0, 5.0],
    ]
}

fn slab_triangles() -> Vec<[u32; 3]> {
    vec![
        [0, 3, 2],
        [0, 2, 1],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ]
}

#[test]
fn sweep_stops_at_the_floor_for_every_convex_probe() {
    let mut world = new_world(static_config());
    world.spawn(BodyDesc::new(ColliderDesc::new(Shape::plane())).mass(0.0));
    let cuboid = down_sweep(
        &mut world,
        "cuboid probe",
        &Shape::cuboid([0.5, 1.5, 0.5]),
        [0.0, 5.0, 0.0],
        20.0,
    );
    assert!(
        (cuboid.distance - 3.5).abs() < 1e-2,
        "a cuboid probe must stop a half height above the plane, got {}",
        cuboid.distance
    );
    let capsule = down_sweep(
        &mut world,
        "capsule probe",
        &Shape::capsule(0.4, 0.6),
        [0.0, 5.0, 0.0],
        20.0,
    );
    assert!(
        (capsule.distance - 4.0).abs() < 1e-2,
        "a capsule probe must stop a half height plus radius above the plane, got {}",
        capsule.distance
    );
    let cylinder = down_sweep(
        &mut world,
        "cylinder probe",
        &Shape::cylinder(0.4, 0.6),
        [0.0, 5.0, 0.0],
        20.0,
    );
    assert!(
        (cylinder.distance - 4.4).abs() < 1e-2,
        "a cylinder probe must stop a half height above the plane, got {}",
        cylinder.distance
    );
}

#[test]
fn filters_skip_each_body_kind() {
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
    let a = world.spawn(
        BodyDesc::static_sphere(0.5)
            .position([0.0, 0.0, 2.0])
            .filter(CollisionFilter::new(1, 1)),
    );
    let b = world.spawn(
        BodyDesc::static_sphere(0.5)
            .position([0.0, 0.0, 4.0])
            .filter(CollisionFilter::new(2, 2)),
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
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
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
fn an_observation_lapses_only_past_the_retention_window() {
    let mut world = new_world(static_config());
    let target = query_static(&mut world, 0.5, [0.0, 0.0, 2.0]);
    let first = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        10.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    assert_eq!(
        world.query_hit(first).map(|hit| hit.body),
        Some(target),
        "first must resolve"
    );
    for _ in 0..dynamis_gpu::FACT_LAG - 1 {
        world.ray_query(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            10.0,
            &QueryFilter::default(),
        );
        world.step(DT);
        world.wait();
    }
    assert_eq!(
        world.query_state(first),
        QueryState::Retired,
        "an observation inside the retention window must stay resolvable"
    );
    assert!(
        world.query_hit(first).is_some(),
        "a retained observation must keep its hits"
    );
    for _ in 0..2 {
        world.ray_query(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            10.0,
            &QueryFilter::default(),
        );
        world.step(DT);
        world.wait();
    }
    assert_eq!(
        world.query_state(first),
        QueryState::Lapsed,
        "an observation past the retention window must lapse"
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.query_hit(first))).is_err(),
        "a lapsed observation must stop resolving"
    );
}

#[test]
fn handle_without_arrived_results_panics() {
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
    let body = query_static(&mut world, 0.5, [0.0, 0.0, 0.0]);
    world.step(DT);
    world.wait();
    let inside = world.point_query([0.1, 0.0, 0.0], &QueryFilter::default());
    world.wait();
    let inside_hit = world.query_hit(inside);
    assert_eq!(inside_hit.map(|hit| hit.body), Some(body));
    let outside = world.point_query([5.0, 0.0, 0.0], &QueryFilter::default());
    world.wait();
    assert!(
        world.query_hit(outside).is_none(),
        "point outside must miss"
    );
    let edge = world.point_query([0.5, 0.0, 0.0], &QueryFilter::default());
    world.wait();
    assert!(
        world.query_hit(edge).is_some(),
        "point on the surface must hit"
    );
}

#[test]
fn overlap_query_accepts_any_convex_shape() {
    let mut world = new_world(static_config());
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
    world.wait();
    let hit = world.query_hit(handle);
    assert_eq!(hit.map(|hit| (hit.body, hit.collider)), Some((body, 1)));
    let miss = world.overlap_query(
        &probe,
        [0.0, 0.0, 0.0, 1.0],
        [8.0, 0.0, 0.0],
        &QueryFilter::default(),
    );
    world.wait();
    assert!(world.query_hit(miss).is_none());
}

#[test]
fn include_filter_limits_results_to_one_body() {
    let mut world = new_world(static_config());
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
    world.wait();
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
    let mut world = new_world(static_config());
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
        world.wait();
        let hit = world
            .query_hit(handle)
            .expect("down sweep must hit the floor");
        assert!(
            (hit.normal[1] - 1.0).abs() < 1e-3,
            "a down sweep over x={x} must report a floor normal facing the sweep, got {:?}",
            hit.normal
        );
    }
}

#[test]
fn capsule_sweep_stops_before_wall_face() {
    let mut world = new_world(static_config());
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
    world.wait();
    let hit = world
        .query_hit(handle)
        .expect("sweep must stop at the wall");
    assert!(
        hit.distance > 0.0 && hit.distance < 0.25,
        "sweep must stop short of the wall face, got {}",
        hit.distance
    );
    assert!(
        hit.normal[0] < -0.99,
        "wall normal must face the sweeping sphere, got {:?}",
        hit.normal
    );
}

#[test]
fn query_after_teleport_sees_the_new_position() {
    let mut world = new_world(static_config());
    let body = query_static(&mut world, 0.5, [0.0, 0.0, 10.0]);

    world.set_position(body, [0.0, 0.0, 2.0]);
    let query = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    world.wait();
    let hit = world
        .query_hit(query)
        .expect("ray must hit the teleported body");
    assert_eq!(hit.body, body);
    assert!((hit.distance - 1.5).abs() < 1e-3, "got {}", hit.distance);
    assert!(!world.query_overflow(query));

    let stale = world.point_query([0.0, 0.0, 10.0], &QueryFilter::default());
    world.wait();
    assert!(
        world.query_hit(stale).is_none(),
        "stale entries must not answer for the old position"
    );
}

#[test]
fn truncated_ray_query_keeps_the_closest_hits() {
    let mut world = new_world(static_config());
    let nearest = query_static(&mut world, 0.5, [0.0, 0.0, 2.0]);
    let middle = query_static(&mut world, 0.5, [0.0, 0.0, 4.0]);
    let _farthest = query_static(&mut world, 0.5, [0.0, 0.0, 6.0]);
    let single_handle = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            max_hits: 1,
            ..QueryFilter::default()
        },
    );
    let pair_handle = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            max_hits: 2,
            ..QueryFilter::default()
        },
    );
    world.step(DT);
    world.wait();
    let single = world.query_hits(single_handle);
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].body, nearest);
    assert!((single[0].distance - 1.5).abs() < 1e-3);
    assert!(world.query_overflow(single_handle));
    let pair = world.query_hits(pair_handle);
    assert_eq!(pair.len(), 2);
    assert_eq!(pair[0].body, nearest);
    assert_eq!(pair[1].body, middle);
    assert!((pair[1].distance - 3.5).abs() < 1e-3);
    assert!(world.query_overflow(pair_handle));
}

#[test]
fn truncated_sweep_query_keeps_the_closest_obstacle() {
    let mut world = new_world(static_config());
    let _near_wall = world.spawn(
        BodyDesc::cuboid([0.25, 3.0, 4.0])
            .mass(0.0)
            .position([2.0, 1.5, 0.0]),
    );
    let _far_wall = world.spawn(
        BodyDesc::cuboid([0.25, 3.0, 4.0])
            .mass(0.0)
            .position([8.0, 1.5, 0.0]),
    );
    world.step(DT);
    world.wait();
    let handle = world.sweep_query(
        &Shape::sphere(0.4),
        [0.0, 0.0, 0.0, 1.0],
        [1.3, 0.95, 0.0],
        [1.0, 0.0, 0.0],
        10.0,
        &QueryFilter {
            max_hits: 1,
            ..QueryFilter::default()
        },
    );
    world.wait();
    let hit = world.query_hit(handle).expect("sweep must hit a wall");
    assert!(
        hit.distance < 1.0,
        "truncated sweep must stop at the nearest wall, got {}",
        hit.distance
    );
}

#[test]
fn exact_hit_count_reports_no_overflow_and_keeps_order() {
    let mut world = new_world(static_config());
    let near = query_static(&mut world, 0.4, [0.0, 0.0, 2.0]);
    let middle = query_static(&mut world, 0.4, [0.0, 0.0, 2.7]);
    let far = query_static(&mut world, 0.4, [0.0, 0.0, 3.4]);
    let exact = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            max_hits: 3,
            ..QueryFilter::default()
        },
    );
    let capped = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            max_hits: 2,
            ..QueryFilter::default()
        },
    );
    world.step(DT);
    world.wait();
    let hits = world.query_hits(exact);
    assert_eq!(
        hits.iter().map(|hit| hit.body).collect::<Vec<_>>(),
        vec![near, middle, far]
    );
    assert!(!world.query_overflow(exact));
    let hits = world.query_hits(capped);
    assert_eq!(
        hits.iter().map(|hit| hit.body).collect::<Vec<_>>(),
        vec![near, middle]
    );
    assert!(world.query_overflow(capped));
}

#[test]
fn queries_do_not_inherit_candidates_from_earlier_queries() {
    let mut world = new_world(static_config());
    let first_target = query_static(&mut world, 0.5, [0.0, 0.0, 2.0]);
    let _second = query_static(&mut world, 0.5, [0.0, 0.0, 4.0]);
    let _third = query_static(&mut world, 0.5, [0.0, 0.0, 6.0]);
    let far_target = query_static(&mut world, 0.5, [10.0, 0.0, 2.0]);
    let first = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            max_hits: 4,
            ..QueryFilter::default()
        },
    );
    world.wait();
    assert_eq!(
        world.query_hit(first).map(|hit| hit.body),
        Some(first_target)
    );
    let second = world.ray_query(
        [10.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter {
            max_hits: 4,
            ..QueryFilter::default()
        },
    );
    world.wait();
    assert_eq!(
        world.query_hit(second).map(|hit| hit.body),
        Some(far_target),
        "a later query must not answer with an earlier query's candidates"
    );
}

#[test]
fn a_query_batch_that_fills_the_stream_keeps_every_result() {
    let mut world = new_world(static_config());
    let target = query_static(&mut world, 0.5, [0.0, 0.0, 2.0]);
    let count = dynamis_domain::STREAM_FLOOR;
    let handles = (0..count)
        .map(|_| {
            world.ray_query(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                20.0,
                &QueryFilter::default(),
            )
        })
        .collect::<Vec<_>>();
    world.step(DT);
    world.wait();
    assert!(
        handles
            .iter()
            .all(|handle| world.query_state(*handle) == QueryState::Retired),
        "a batch as wide as the query stream must resolve completely"
    );
    assert!(
        handles
            .iter()
            .all(|handle| world.query_hit(*handle).map(|hit| hit.body) == Some(target)),
        "every query of a full batch must report the body it hits"
    );
}

#[test]
fn a_query_run_preserves_pending_step_inputs() {
    let mut world = new_world(static_config());
    let _ground = static_sphere_ground(&mut world, 1.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 4.0, 0.0]));
    settle(&mut world, 2);
    let before = world.read_state(ball).velocity;
    world.apply_force(ball, [0.0, 600.0, 0.0]);
    world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        20.0,
        &QueryFilter::default(),
    );
    world.resolve_queries();
    world.step(DT);
    world.wait();
    let after = world.read_state(ball).velocity;
    let rise = after[1] - before[1];
    assert!(
        (rise - 600.0 / 60.0).abs() < 1e-2,
        "a force applied before a query resolve must still accelerate the body, rose {rise}"
    );
}

#[test]
fn a_query_run_declares_the_passes_it_runs() {
    let mut world = new_world(static_config());
    let _target = query_static(&mut world, 0.5, [0.0, 0.0, 2.0]);
    settle(&mut world, 2);
    world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    world.resolve_queries();
    let ran = world.ran_passes();
    for pass in ["commands", "query_aabbs", "entries", "broadphase", "query"] {
        assert!(
            ran.contains(&pass),
            "a query resolve must run {pass}, ran {ran:?}"
        );
    }
    assert!(
        !ran.contains(&"substeps"),
        "a query resolve must not advance the simulation, ran {ran:?}"
    );
}

#[test]
fn a_query_run_leaves_pending_inputs_for_the_next_step_once() {
    let mut world = new_world(static_config());
    let _ground = static_sphere_ground(&mut world, 1.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 4.0, 0.0]));
    settle(&mut world, 2);
    let before = world.read_state(ball).velocity;
    world.apply_impulse(ball, [60.0, 0.0, 0.0]);
    world.point_query([0.0, 4.0, 0.0], &QueryFilter::default());
    world.resolve_queries();
    world.step(DT);
    world.wait();
    let after = world.read_state(ball).velocity;
    let gained = after[0] - before[0];
    assert!(
        (gained - 60.0).abs() < 1e-2,
        "an impulse applied before a query resolve must land exactly once, gained {gained}"
    );
}

#[test]
fn a_query_run_wakes_a_sleeping_body_for_its_pending_force() {
    let mut world = new_world(gravity_config());
    let _floor = static_sphere_ground(&mut world, 1.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.5, 0.0]));
    settle_until(&mut world, 600, |world| world.read_state(ball).sleeping);
    world.apply_force(ball, [0.0, 600.0, 0.0]);
    world.ray_query(
        [0.0, 4.0, 0.0],
        [0.0, -1.0, 0.0],
        20.0,
        &QueryFilter::default(),
    );
    world.resolve_queries();
    world.step(DT);
    world.wait();
    let state = world.read_state(ball);
    let expected = 600.0 / 60.0 - 9.81 / 60.0;
    assert!(
        !state.sleeping,
        "the pending force must wake the body again"
    );
    assert!(
        (state.velocity[1] - expected).abs() < 1e-2,
        "a sleeping body must still receive the force a query resolve left pending, got {:?}",
        state.velocity
    );
}

#[test]
fn a_patch_survives_both_a_query_run_and_a_step() {
    let mut world = new_world(static_config());
    let body = query_static(&mut world, 0.5, [0.0, 0.0, 10.0]);
    world.set_position(body, [0.0, 0.0, 2.0]);
    world.point_query([0.0, 0.0, 2.0], &QueryFilter::default());
    world.resolve_queries();
    world.step(DT);
    world.wait();
    assert_eq!(world.read_state(body).position, [0.0, 0.0, 2.0]);
    let query = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    world.wait();
    let hit = world
        .query_hit(query)
        .expect("the patched body must stay put");
    assert_eq!(hit.body, body);
    assert!((hit.distance - 1.5).abs() < 1e-3, "got {}", hit.distance);
}

#[test]
fn a_query_wider_than_the_walk_budget_still_finds_a_distant_body() {
    let mut world = new_world(static_config());
    let near = query_static(&mut world, 0.02, [0.0, 0.0, 0.0]);
    let far = query_static(&mut world, 0.02, [0.0, 0.0, 40.0]);
    let query = world.sphere_query([0.0, 0.0, 0.0], 100.0, &QueryFilter::default());
    world.step(DT);
    world.wait();
    assert!(
        !world.query_overflow(query),
        "a single cell per body must fit the candidate budget"
    );
    let hits = world.query_hits(query);
    assert!(
        hits.iter().any(|hit| hit.body == near) && hits.iter().any(|hit| hit.body == far),
        "a box spanning far more cells than the walk visits must fall back to its level and still find every body"
    );
}

#[test]
fn a_down_sweep_measures_the_same_gap_anywhere_on_a_large_floor() {
    let mut world = new_world(static_config());
    world.spawn(
        BodyDesc::cuboid([60.0, 0.5, 60.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    for start in [
        [0.0, 0.55, 0.0],
        [0.8, 0.55, 1.2],
        [20.0, 0.55, 20.0],
        [-30.0, 0.55, 12.5],
        [59.0, 0.55, -59.0],
    ] {
        let hit = down_sweep(&mut world, "large floor", &Shape::sphere(0.35), start, 0.4);
        assert!(
            (hit.normal[1] - 1.0).abs() < 1e-3,
            "a large floor must report a floor normal from {start:?}, got {:?}",
            hit.normal
        );
        assert!(
            (hit.distance - 0.2).abs() < 1e-3,
            "a large floor must report the same gap from {start:?}, got {}",
            hit.distance
        );
    }
}
