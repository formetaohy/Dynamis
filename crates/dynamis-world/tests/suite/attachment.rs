use super::common::{DT, distance, gravity_config, new_world, settle};
use dynamis_model::{BodyDesc, BodyHandle, ColliderDesc, Shape, SoftAttachment, SoftBodyDesc};
use dynamis_world::World;

fn sensor_anchor(world: &mut World, position: [f32; 3]) -> BodyHandle {
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.1)).sensor(true))
            .mass(0.0)
            .position(position),
    )
}

fn hanging_chain(
    world: &mut World,
    anchor: BodyHandle,
    top: [f32; 3],
) -> dynamis_model::SoftBodyHandle {
    world.add_soft_body(
        SoftBodyDesc::net(
            vec![
                [0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, -3.0, 0.0],
            ],
            vec![[0, 1], [1, 2], [2, 3]],
        )
        .radius(0.1)
        .position(top)
        .attach(SoftAttachment::new(0, anchor, [0.0; 3])),
    )
}

#[test]
fn an_attachment_holds_a_soft_body_to_its_anchor() {
    let mut world = new_world(gravity_config());
    let anchor = sensor_anchor(&mut world, [0.0, 8.0, 0.0]);
    let soft = hanging_chain(&mut world, anchor, [0.0, 8.0, 0.0]);
    settle(&mut world, 180);
    let positions = world.inspect_soft_particles(soft);
    assert!(
        distance(positions[0], [0.0, 8.0, 0.0]) < 0.01,
        "the attached particle must hold its anchor, got {:?}",
        positions[0]
    );
    assert!(
        (positions[3][1] - 5.0).abs() < 0.05 && positions[3][0].abs() < 0.05,
        "the chain must hang below its anchor, got {:?}",
        positions[3]
    );
}

#[test]
fn an_attachment_carries_a_soft_body_with_a_moving_anchor() {
    let mut world = new_world(gravity_config());
    let carrier = world.spawn(
        BodyDesc::cuboid([0.1; 3])
            .kinematic(true)
            .position([0.0, 5.0, 0.0]),
    );
    let soft = world.add_soft_body(
        SoftBodyDesc::net(
            vec![[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -2.0, 0.0]],
            vec![[0, 1], [1, 2]],
        )
        .radius(0.1)
        .position([0.0, 5.0, 0.0])
        .attach(SoftAttachment::new(0, carrier, [0.0; 3])),
    );
    for frame in 0..60 {
        let x = frame as f32 * DT * 2.0;
        world.set_position(carrier, [x, 5.0, 0.0]);
        world.set_velocity(carrier, [2.0, 0.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let positions = world.inspect_soft_particles(soft);
    assert!(
        (positions[0][0] - 2.0).abs() < 0.1,
        "the attached particle must follow its anchor, got {:?}",
        positions[0]
    );
    assert!(
        positions[2][0] > 1.0,
        "the carried soft body must trail its anchor, got {:?}",
        positions[2]
    );
}

#[test]
fn a_moving_anchor_wakes_a_sleeping_soft_body() {
    let mut world = new_world(gravity_config());
    let carrier = world.spawn(
        BodyDesc::cuboid([0.1; 3])
            .kinematic(true)
            .position([0.0, 5.0, 0.0]),
    );
    let soft = world.add_soft_body(
        SoftBodyDesc::net(vec![[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]], vec![[0, 1]])
            .radius(0.1)
            .position([0.0, 5.0, 0.0])
            .attach(SoftAttachment::new(0, carrier, [0.0; 3])),
    );
    settle(&mut world, 180);
    for frame in 0..40 {
        let x = 3.0 + frame as f32 * DT * 2.0;
        world.set_position(carrier, [x, 5.0, 0.0]);
        world.set_velocity(carrier, [2.0, 0.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let positions = world.inspect_soft_particles(soft);
    assert!(
        positions[0][0] > 2.5,
        "a sleeping soft body must wake when its anchor moves, got {:?}",
        positions[0]
    );
}

#[test]
fn a_restored_attachment_still_anchors_its_soft_body() {
    let mut world = new_world(gravity_config());
    let anchor = sensor_anchor(&mut world, [0.0, 8.0, 0.0]);
    let soft = hanging_chain(&mut world, anchor, [0.0, 8.0, 0.0]);
    settle(&mut world, 60);
    let snapshot = world.snapshot();
    settle(&mut world, 30);
    world.restore(&snapshot);
    settle(&mut world, 60);
    let positions = world.inspect_soft_particles(soft);
    assert!(
        distance(positions[0], [0.0, 8.0, 0.0]) < 0.01,
        "a restored attachment must still hold its anchor, got {:?}",
        positions[0]
    );
    assert!(
        (positions[3][1] - 5.0).abs() < 0.05,
        "a restored soft body must still hang from its anchor, got {:?}",
        positions[3]
    );
}

#[test]
fn an_attachment_follows_its_anchor_through_a_row_move() {
    let mut world = new_world(gravity_config());
    let neighbour = world.spawn(BodyDesc::cuboid([0.1; 3]).position([9.0, 0.0, 0.0]));
    let anchor = sensor_anchor(&mut world, [0.0, 8.0, 0.0]);
    let soft = hanging_chain(&mut world, anchor, [0.0, 8.0, 0.0]);
    settle(&mut world, 30);
    world.remove(neighbour);
    settle(&mut world, 90);
    let positions = world.inspect_soft_particles(soft);
    assert!(
        distance(positions[0], [0.0, 8.0, 0.0]) < 0.01,
        "an attachment must resolve its anchor row after a scene swap, got {:?}",
        positions[0]
    );
    assert!(
        (positions[3][1] - 5.0).abs() < 0.05,
        "the chain must keep hanging below its anchor, got {:?}",
        positions[3]
    );
}

#[test]
fn removing_a_soft_body_releases_its_anchors() {
    let mut world = new_world(gravity_config());
    let anchor = sensor_anchor(&mut world, [0.0, 8.0, 0.0]);
    let soft = hanging_chain(&mut world, anchor, [0.0, 8.0, 0.0]);
    world.remove_soft_body(soft);
    world.remove(anchor);
    settle(&mut world, 10);
}

#[test]
#[should_panic(expected = "still anchors a soft attachment")]
fn a_body_that_still_anchors_a_soft_body_refuses_removal() {
    let mut world = new_world(gravity_config());
    let anchor = sensor_anchor(&mut world, [0.0, 8.0, 0.0]);
    let _soft = hanging_chain(&mut world, anchor, [0.0, 8.0, 0.0]);
    world.remove(anchor);
}
