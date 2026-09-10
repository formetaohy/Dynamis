use super::common::{DT, settle, sim, static_config, static_sphere_ground};
use dynamis_layout::{COUNTER_CONTACTS, COUNTER_SPILLOVER_PAIRS};
use dynamis_model::{BodyDesc, BodyHandle};
use dynamis_simulate::Simulation;

const PILE: usize = 320;
const LATTICE: usize = 7;

fn sphere_pile(world: &mut Simulation) -> Vec<BodyHandle> {
    (0..PILE)
        .map(|index| {
            let position = [
                (index % LATTICE) as f32,
                (index / LATTICE % LATTICE) as f32,
                (index / (LATTICE * LATTICE)) as f32,
            ];
            world.spawn(BodyDesc::static_sphere(6.0).position(position))
        })
        .collect()
}

#[test]
fn a_pile_heavier_than_the_streams_widens_them_until_the_step_stops_spilling() {
    let mut world = sim(PILE, static_config());
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

fn step_waited(world: &mut Simulation, steps: usize) {
    for _ in 0..steps {
        world.step(DT);
        world.wait();
    }
}

#[test]
fn sustained_idleness_releases_the_widened_streams_without_starving_the_next_scene() {
    let mut world = sim(PILE, static_config());
    let bodies = sphere_pile(&mut world);
    step_waited(&mut world, 4);
    let widened = world.stream_capacity();

    for body in bodies {
        world.remove(body);
    }
    step_waited(&mut world, 140);
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
