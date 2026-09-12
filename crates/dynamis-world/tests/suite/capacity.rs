use super::common::{
    DT, flat_mesh_floor, gravity_config, new_world, settle, settle_until, static_config,
    static_sphere_ground,
};
use dynamis_layout::{COUNTER_CONTACTS, COUNTER_SPILLOVER_PAIRS};
use dynamis_model::{BodyDesc, BodyHandle, ColliderDesc, QueryFilter, Shape};
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
fn a_pile_heavier_than_the_streams_widens_them_until_the_step_stops_spilling() {
    let mut world = new_world(static_config());
    let floor = world.stream_capacity();
    sphere_pile(&mut world);

    world.step(DT);
    world.wait();
    assert!(
        world.measured()[COUNTER_SPILLOVER_PAIRS] > 0,
        "the pile must outgrow the pair stream"
    );
    let planned = world.stream_capacity();
    assert!(
        planned.pairs > floor.pairs,
        "the live rows alone must widen the plan"
    );

    world.step(DT);
    world.wait();
    assert!(
        world.stream_capacity().pairs > planned.pairs,
        "the spilled step must widen the plan further"
    );
    assert_eq!(
        world.measured()[COUNTER_SPILLOVER_PAIRS],
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
        released.pairs < widened.pairs,
        "an idle peak must be released, {released:?} vs {widened:?}"
    );

    static_sphere_ground(&mut world, 1.0);
    world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.5, 0.0]));
    settle(&mut world, 3);
    assert_eq!(
        world.measured()[COUNTER_SPILLOVER_PAIRS],
        0,
        "the released stream must still serve a small scene"
    );
    assert!(
        world.measured()[COUNTER_CONTACTS] > 0,
        "the world must still resolve contacts"
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
        released.pairs < widened.pairs && released.entries < widened.entries,
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
    let states = world.state_buffer().token();
    let pairs = world.stream_capacity().pairs;
    assert_eq!(
        world.measured()[COUNTER_SPILLOVER_PAIRS],
        0,
        "a spread pile must fit the pair stream"
    );
    for (index, body) in bodies.iter().enumerate() {
        world.set_position(*body, spread_position(index, 1.0));
    }
    settle_until(&mut world, 120, |world| {
        world.stream_capacity().pairs > pairs
    });
    assert!(
        world.stream_capacity().pairs > pairs,
        "the crowded pile must widen the pair stream"
    );
    assert_eq!(
        world.state_buffer().token(),
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
        "the scene must keep integrating across the transition"
    );
}

fn floor_hit_height(world: &mut World) -> Option<f32> {
    let handle = world.ray_query(
        [5.0, 3.0, 5.0],
        [0.0, -1.0, 0.0],
        10.0,
        &QueryFilter::default(),
    );
    world.flush_queries();
    world.query_hit(handle).map(|hit| hit.point[1])
}

#[test]
fn a_shape_stream_swap_keeps_uploaded_geometry() {
    let mut world = new_world(gravity_config());
    flat_mesh_floor(&mut world);
    settle(&mut world, 2);
    let plan = world.stream_capacity();
    let before = floor_hit_height(&mut world).expect("the floor must answer a downward ray");

    let (vertices, triangles) = lifted_mesh();
    let lifted = world.add_mesh(&vertices, &triangles);
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::mesh(lifted)))
            .mass(0.0)
            .position([0.0, 500.0, 0.0]),
    );
    settle(&mut world, 2);

    assert!(
        world.stream_capacity().shapes.vertices > plan.shapes.vertices,
        "the lifted mesh must widen the shape streams"
    );
    let after = floor_hit_height(&mut world)
        .expect("the uploaded floor must stay visible to queries across the shape stream swap");
    assert!(
        (after - before).abs() < 1e-3,
        "the uploaded floor must keep its height, {before} became {after}"
    );
}
