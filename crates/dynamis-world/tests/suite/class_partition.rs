use super::common::{DT, gravity_config, new_world};
use dynamis_layout::COUNTER_CLASS_CONFLICTS;
use dynamis_model::{BodyDesc, BodyHandle};
use dynamis_world::World;

fn dense_pile(world: &mut World) -> Vec<BodyHandle> {
    world.spawn(
        BodyDesc::cuboid([6.0, 0.5, 6.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let mut bodies = Vec::new();
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                bodies.push(world.spawn(BodyDesc::sphere(0.6).position([
                    (x as f32 - 1.5) * 0.9,
                    0.6 + y as f32 * 0.9,
                    (z as f32 - 1.5) * 0.9,
                ])));
            }
        }
    }
    bodies
}

fn assert_conflict_free(world: &World, step: usize) {
    let conflicts = world.measured()[COUNTER_CLASS_CONFLICTS];
    assert_eq!(
        conflicts, 0,
        "step {step} ran {conflicts} solver blocks of one body in the same class"
    );
}

#[test]
fn solver_class_partition_never_joins_blocks_of_one_body() {
    let mut world = new_world(gravity_config());
    let bodies = dense_pile(&mut world);
    for step in 0..240 {
        world.step(DT);
        world.wait();
        assert_conflict_free(&world, step);
    }
    for (index, body) in bodies.iter().enumerate() {
        if index % 3 == 0 {
            world.remove(*body);
        }
    }
    for step in 0..120 {
        world.step(DT);
        world.wait();
        assert_conflict_free(&world, step);
    }
}
