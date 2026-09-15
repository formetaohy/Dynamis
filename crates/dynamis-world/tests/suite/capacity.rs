use super::common::{
    DT, flat_mesh_floor, gravity_config, new_world, settle, settle_until, static_config,
    static_sphere_ground,
};
use dynamis_abi::{
    COUNTER_COLLIDERS, COUNTER_CONTACTS, COUNTER_EVENTS, COUNTER_REFUSED_CONTACTS,
    COUNTER_REFUSED_PAIRS, COUNTER_RESTING,
};
use dynamis_model::{BodyDesc, BodyHandle, ColliderDesc, QueryFilter, Shape, SoftBodyDesc};
use dynamis_world::World;

const PILE: usize = 320;
const LATTICE: usize = 7;

fn sphere_pile(world: &mut World) -> Vec<BodyHandle> {
    (0..PILE)
        .map(|index| {
            let position = [
                (index % LATTICE) as f32,
                (index / LATTICE % LATTICE) as f32,
                (index / (LATTICE * LATTICE)) as f32,
            ];
            world.spawn(BodyDesc::sphere(6.0).position(position))
        })
        .collect()
}

#[test]
fn the_rigid_shape_follows_the_live_scene_while_the_streams_hold_a_peak() {
    let mut world = new_world(static_config());
    let floor = world.stream_capacity();
    let quiet = world.rigid_shape();

    let pile = sphere_pile(&mut world);
    settle(&mut world, 4);
    let crowded = world.rigid_shape();
    assert!(
        world.stream_capacity().broadphase.pairs > floor.broadphase.pairs,
        "the pile must widen the pair stream"
    );
    assert!(
        crowded.island_rounds > quiet.island_rounds,
        "a wider island must propagate over more rounds, {quiet:?} vs {crowded:?}"
    );
    assert!(
        crowded.body_row_words > quiet.body_row_words,
        "more live rows must widen the key digits, {quiet:?} vs {crowded:?}"
    );

    for body in pile {
        world.remove(body);
    }
    assert!(
        world.stream_capacity().broadphase.pairs > floor.broadphase.pairs,
        "removing the pile must leave the widened streams allocated"
    );
    let shaped = world.rigid_shape();
    assert_eq!(
        (
            shaped.island_rounds,
            shaped.body_row_words,
            shaped.collider_slot_words,
        ),
        (
            quiet.island_rounds,
            quiet.body_row_words,
            quiet.collider_slot_words,
        ),
        "an allocated peak may not keep shaping a step whose live rows are gone"
    );
    assert_eq!(
        shaped.body_id_words, crowded.body_id_words,
        "the body id space outlives every row it once held"
    );
}

fn cloth(side: usize) -> SoftBodyDesc {
    let mut positions = Vec::new();
    for row in 0..side {
        for col in 0..side {
            positions.push([col as f32 * 0.1, 0.0, row as f32 * 0.1]);
        }
    }
    let mut links = Vec::new();
    for row in 0..side {
        for col in 0..side {
            let here = (row * side + col) as u32;
            if col + 1 < side {
                links.push([here, here + 1]);
            }
            if row + 1 < side {
                links.push([here, here + side as u32]);
            }
        }
    }
    SoftBodyDesc::net(positions, links)
        .radius(0.02)
        .position([0.0, 1.0, 0.0])
}

#[test]
fn a_domain_plans_from_its_own_live_data_alone() {
    let mut soft_world = new_world(static_config());
    let soft_floor = soft_world.stream_capacity();
    soft_world.add_soft_body(cloth(10));
    settle(&mut soft_world, 8);
    let soft = soft_world.stream_capacity();
    assert!(
        soft.soft.particles > soft_floor.soft.particles,
        "a soft body must widen its own particle stream, {soft:?} vs {soft_floor:?}"
    );
    assert_eq!(
        soft.rigid, soft_floor.rigid,
        "a soft body may not shape the rigid streams"
    );
    assert_eq!(
        soft.state, soft_floor.state,
        "a soft body may not shape the state streams"
    );

    let mut rigid_world = new_world(static_config());
    let rigid_floor = rigid_world.stream_capacity();
    for index in 0..48 {
        rigid_world.spawn(BodyDesc::sphere(6.0).position([
            (index % 4) as f32,
            (index / 4 % 3) as f32,
            (index / 12) as f32,
        ]));
    }
    settle(&mut rigid_world, 2);
    let rigid = rigid_world.stream_capacity();
    assert!(
        rigid.broadphase.pairs > rigid_floor.broadphase.pairs,
        "a body pile must widen its own pair stream, {rigid:?} vs {rigid_floor:?}"
    );
    assert_eq!(
        rigid.soft, rigid_floor.soft,
        "a body pile may not shape the soft streams"
    );
}

#[test]
fn a_pile_heavier_than_the_streams_widens_them_until_the_step_stops_spilling() {
    let mut world = new_world(static_config());
    let floor = world.stream_capacity();
    sphere_pile(&mut world);

    world.step(DT);
    let served = world.stream_capacity();
    assert!(
        served.broadphase.pairs > floor.broadphase.pairs,
        "the live rows alone must widen the plan"
    );
    world.wait();
    assert!(
        world.measured()[COUNTER_REFUSED_PAIRS] > 0,
        "the pile must outgrow the pair stream"
    );
    assert!(
        world.stream_capacity().broadphase.pairs > served.broadphase.pairs,
        "the spilled step must widen the plan further"
    );

    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_REFUSED_PAIRS],
        0,
        "the widened stream must serve the pile"
    );
}

#[test]
fn sustained_idleness_releases_the_widened_streams_without_starving_the_next_scene() {
    let mut world = new_world(static_config());
    let bodies = sphere_pile(&mut world);
    for _ in 0..4 {
        world.step(DT);
        world.wait();
    }
    let widened = world.stream_capacity();

    for body in bodies {
        world.remove(body);
    }
    for _ in 0..140 {
        world.step(DT);
        world.wait();
    }
    let released = world.stream_capacity();
    assert!(
        released.broadphase.pairs < widened.broadphase.pairs,
        "an idle peak must be released, {released:?} vs {widened:?}"
    );

    static_sphere_ground(&mut world, 1.0);
    world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.5, 0.0]));
    settle(&mut world, 3);
    assert_eq!(
        world.measured()[COUNTER_REFUSED_PAIRS],
        0,
        "the released stream must still serve a small world"
    );
    assert!(
        world.measured()[COUNTER_CONTACTS] > 0,
        "the world must still resolve contacts"
    );
}

#[test]
fn a_kept_reservation_serves_a_whole_scene_without_a_second_allocation() {
    let mut world = new_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([6.0, 0.5, 6.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let ball = world.spawn(
        BodyDesc::sphere(0.4)
            .position([0.0, 0.4, 0.0])
            .friction(0.8)
            .sleep_thresholds(0.0, 0.0),
    );
    settle(&mut world, 60);
    assert!(
        !world.read_state(ball).sleeping,
        "the scene must keep simulating while the reservation is held"
    );

    let settled = world.stream_capacity();
    let submissions = world.submissions();
    for _ in 0..120 {
        world.step(DT);
    }

    assert_eq!(
        world.stream_capacity(),
        settled,
        "a world whose demand stays within its reservation must keep its streams"
    );
    assert_eq!(
        world.submissions() - submissions,
        120,
        "a kept reservation must let a step be exactly one submission"
    );
}

#[test]
fn a_widening_scene_reaches_its_reservation_by_doubling() {
    let mut world = new_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([9.0, 0.5, 9.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let floor = world.stream_capacity();
    sphere_pile(&mut world);
    settle(&mut world, 2);

    let mut allocations = 0;
    let mut previous = world.submissions();
    for _ in 0..240 {
        world.step(DT);
        let submissions = world.submissions();
        if submissions - previous != 1 {
            allocations += 1;
        }
        previous = submissions;
    }

    assert!(
        allocations <= 8,
        "a scene that widens its demand by orders of magnitude must reach its reservation in doublings, not in {allocations} allocations"
    );
    assert!(
        world.stream_capacity().broadphase.pairs > floor.broadphase.pairs,
        "the explosion must widen the pair stream"
    );
}

#[test]
fn the_declared_collider_fact_counts_the_live_colliders_alone() {
    let mut world = new_world(static_config());
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_COLLIDERS],
        0,
        "an empty world declares no collider"
    );

    let compound = world.spawn(
        BodyDesc::sphere(0.2)
            .position([0.0, 0.0, 0.0])
            .collider(ColliderDesc::new(Shape::sphere(0.3)))
            .collider(ColliderDesc::new(Shape::capsule(0.2, 0.1))),
    );
    world.spawn(BodyDesc::sphere(0.5).position([10.0, 0.0, 0.0]));
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_COLLIDERS],
        4,
        "a compound body declares every collider it holds"
    );

    world.remove(compound);
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_COLLIDERS],
        1,
        "a removed body releases the colliders it held"
    );
}

#[test]
fn an_idle_scene_sheds_the_room_its_colliders_never_used() {
    let mut world = new_world(static_config());
    for index in 0..512 {
        world.spawn(BodyDesc::static_sphere(0.2).position([index as f32 * 4.0, 0.0, 0.0]));
    }
    settle(&mut world, 140);
    let measured = *world.measured();
    let capacity = world.stream_capacity();
    assert_eq!(
        measured[COUNTER_CONTACTS], 0,
        "spaced colliders must never touch"
    );
    assert_eq!(
        measured[COUNTER_EVENTS], 0,
        "spaced colliders must never announce"
    );
    assert!(
        capacity.rigid.contacts < 512 * 4,
        "an idle scene must shed the contact room its colliders never used, {capacity:?}"
    );
    assert!(
        capacity.rigid.events < 512 * 8,
        "an idle scene must shed the event room its colliders never used, {capacity:?}"
    );
}

#[test]
fn a_narrowed_world_still_resolves_recycled_body_identities() {
    let mut world = new_world(static_config());
    let pile = sphere_pile(&mut world);
    for _ in 0..3 {
        world.step(DT);
        world.wait();
    }
    let widened = world.stream_capacity();
    for body in pile {
        world.remove(body);
    }
    for _ in 0..480 {
        world.step(DT);
        world.wait();
    }
    let released = world.stream_capacity();
    assert!(
        released.broadphase.pairs < widened.broadphase.pairs
            && released.broadphase.entries < widened.broadphase.entries,
        "an idle world must release its widened streams, {released:?} vs {widened:?}"
    );

    world.drain_events();
    static_sphere_ground(&mut world, 1.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.5, 0.0]));
    settle_until(&mut world, 600, |world| world.read_state(ball).sleeping);
    for _ in 0..5 {
        world.step(DT);
        world.wait();
    }
    let ends = world
        .drain_events()
        .into_iter()
        .filter(|event| event.kind == dynamis_model::ContactEventKind::End)
        .count();
    assert_eq!(
        ends, 0,
        "a resting pair must not emit an end event after the streams narrowed"
    );
}

#[test]
fn widening_one_stream_leaves_the_other_streams_allocated() {
    let mut world = new_world(static_config());
    let bodies = (0..PILE)
        .map(|index| world.spawn(BodyDesc::sphere(6.0).position(spread_position(index, 20.0))))
        .collect::<Vec<_>>();
    settle(&mut world, 4);
    let states = world.state_buffer().allocation();
    let pairs = world.stream_capacity().broadphase.pairs;
    assert_eq!(
        world.measured()[COUNTER_REFUSED_PAIRS],
        0,
        "a spread pile must fit the pair stream"
    );
    for (index, body) in bodies.iter().enumerate() {
        world.set_position(*body, spread_position(index, 1.0));
    }
    settle_until(&mut world, 120, |world| {
        world.stream_capacity().broadphase.pairs > pairs
    });
    assert!(
        world.stream_capacity().broadphase.pairs > pairs,
        "the crowded pile must widen the pair stream"
    );
    assert_eq!(
        world.state_buffer().allocation(),
        states,
        "widening a stream must leave the unrelated streams allocated"
    );
}

fn spread_position(index: usize, spacing: f32) -> [f32; 3] {
    [
        (index % LATTICE) as f32 * spacing,
        (index / LATTICE % LATTICE) as f32 * spacing,
        (index / (LATTICE * LATTICE)) as f32 * spacing,
    ]
}

fn distant_static_burst(world: &mut World, count: usize) {
    for index in 0..count {
        world.spawn(BodyDesc::static_sphere(0.05).position([1000.0 + index as f32, 0.0, 0.0]));
    }
}

fn lifted_mesh() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let side = 12u32;
    let mut vertices = Vec::new();
    for row in 0..=side {
        for col in 0..=side {
            vertices.push([col as f32 - 6.0, 0.0, row as f32 - 6.0]);
        }
    }
    let mut triangles = Vec::new();
    for row in 0..side {
        for col in 0..side {
            let a = row * (side + 1) + col;
            triangles.push([a, a + side + 1, a + 1]);
            triangles.push([a + side + 1, a + side + 2, a + 1]);
        }
    }
    (vertices, triangles)
}

#[test]
fn a_plan_transition_lands_the_edits_of_its_own_frame() {
    let mut world = new_world(static_config());
    let ball = world.spawn(
        BodyDesc::sphere(0.3)
            .position([0.0, 0.0, 0.0])
            .velocity([1.0, 0.0, 0.0]),
    );
    settle(&mut world, 4);
    let before = world.read_state(ball).position[0];

    let plan = world.stream_capacity();
    distant_static_burst(&mut world, 160);
    world.set_velocity(ball, [4.0, 0.0, 0.0]);
    world.step(DT);
    world.wait();

    assert!(
        world.stream_capacity() != plan,
        "the burst must widen the plan in the transition frame"
    );
    let state = world.read_state(ball);
    assert_eq!(
        state.velocity,
        [4.0, 0.0, 0.0],
        "an edit of the transition frame must reach the swapped streams"
    );
    assert!(
        state.position[0] > before,
        "the world must keep integrating across the transition"
    );
}

fn floor_hit_height(world: &mut World) -> Option<f32> {
    let handle = world.ray_query(
        [5.0, 3.0, 5.0],
        [0.0, -1.0, 0.0],
        10.0,
        &QueryFilter::default(),
    );
    world.wait();
    world.query_hit(handle).map(|hit| hit.point[1])
}

#[test]
fn a_shape_stream_swap_keeps_uploaded_geometry() {
    let mut world = new_world(gravity_config());
    flat_mesh_floor(&mut world);
    settle(&mut world, 2);
    let plan = world.stream_capacity().state.shapes;
    let before = floor_hit_height(&mut world).expect("the floor must answer a downward ray");

    let (vertices, triangles) = lifted_mesh();
    let lifted = world.add_mesh(&vertices, &triangles, None);
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::mesh(lifted)))
            .mass(0.0)
            .position([0.0, 500.0, 0.0]),
    );
    settle(&mut world, 2);

    assert!(
        world.stream_capacity().state.shapes.vertices > plan.vertices,
        "the lifted mesh must widen the shape streams"
    );
    let after = floor_hit_height(&mut world)
        .expect("the uploaded floor must stay visible to queries across the shape stream swap");
    assert!(
        (after - before).abs() < 1e-3,
        "the uploaded floor must keep its height, {before} became {after}"
    );
}

#[test]
fn the_contact_store_follows_the_contacts_not_the_swept_candidates() {
    let mut world = new_world(static_config());
    let floor = world.stream_capacity();
    let sweepers = (0..512)
        .map(|index| {
            world.spawn(
                BodyDesc::sphere(0.2)
                    .position([0.0, 1.0 + index as f32, 0.0])
                    .velocity([0.0, -400.0, 0.0]),
            )
        })
        .collect::<Vec<_>>();
    settle(&mut world, 2);
    let sweeping = world.stream_capacity();
    assert!(
        sweeping.rigid.pairs > floor.rigid.pairs,
        "a fast column must widen the candidate store, {floor:?} vs {sweeping:?}"
    );
    assert_eq!(
        world.measured()[COUNTER_CONTACTS],
        0,
        "a column spaced wider than its swept reach must touch nothing"
    );
    assert!(
        sweeping.rigid.contacts * 2 < sweeping.rigid.pairs,
        "a world without contacts must keep the contact store far below the candidate store, {sweeping:?}"
    );
    for body in sweepers {
        world.remove(body);
    }

    let pile = sphere_pile(&mut world);
    settle_until(&mut world, 60, |world| {
        world.measured()[COUNTER_REFUSED_CONTACTS] == 0
            && world.stream_capacity().rigid.contacts >= 2 * world.measured()[COUNTER_CONTACTS]
    });
    assert_eq!(
        world.measured()[COUNTER_REFUSED_CONTACTS],
        0,
        "the widened contact store must serve every contact of the pile"
    );
    assert!(
        world.stream_capacity().rigid.contacts > sweeping.rigid.contacts,
        "the spilled pile must widen the contact store"
    );
    let _ = pile;
}

#[test]
fn the_resting_pool_never_shrinks_below_its_watermark() {
    let mut world = new_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let pile = (0..64)
        .map(|index| {
            let x = (index % 8) as f32 * 0.85 - 3.0;
            let z = (index / 8) as f32 * 0.85 - 3.0;
            world.spawn(BodyDesc::sphere(0.4).position([x, 0.4, z]).friction(0.6))
        })
        .collect::<Vec<_>>();
    settle_until(&mut world, 600, |world| {
        world.measured()[COUNTER_RESTING] >= 24
    });
    let frozen = world.stream_capacity().rigid.resting;
    assert!(
        frozen >= world.measured()[COUNTER_RESTING],
        "the resting pool must cover the slots the device froze, {frozen} vs {}",
        world.measured()[COUNTER_RESTING]
    );

    for _ in 0..400 {
        world.step(DT);
        world.wait();
    }
    let idle = world.stream_capacity().rigid;
    assert!(
        idle.resting >= frozen,
        "the frozen slots outlive the contacts that produced them, {frozen} vs {idle:?}"
    );
    assert!(
        world.inspect_contacts().len() >= 24,
        "the released contact store must still report the frozen contacts"
    );
    for body in pile {
        world.remove(body);
    }
}
