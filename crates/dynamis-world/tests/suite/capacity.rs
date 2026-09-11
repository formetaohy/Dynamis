use super::common::{DT, new_world, settle, settle_until, static_config, static_sphere_ground};
use dynamis_layout::{COUNTER_CONTACTS, COUNTER_SPILLOVER_PAIRS};
use dynamis_model::{BodyDesc, BodyHandle};
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
