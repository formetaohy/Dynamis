use super::common::{DT, distance, gravity_config, observed_world, settle, static_config};
use dynamis_model::{BodyDesc, ColliderDesc, Shape, SoftBodyDesc};

fn chain(position: [f32; 3]) -> SoftBodyDesc {
    SoftBodyDesc::net(
        vec![
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        vec![[0, 1], [1, 2], [2, 3]],
    )
    .radius(0.1)
    .position(position)
}

fn anchor(world: &mut dynamis_world::World, position: [f32; 3]) -> dynamis_model::BodyHandle {
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.1)).sensor(true))
            .mass(0.0)
            .position(position),
    )
}

#[test]
fn a_placed_soft_body_keeps_the_pose_it_was_placed_at() {
    let mut world = observed_world(static_config());
    let handle = world.add_soft_body(chain([0.0, 6.0, 0.0]));
    settle(&mut world, 30);
    let before = world.inspect_soft_particles(handle);
    for (particle, position) in before.iter().enumerate() {
        world.set_soft_particle_position(
            handle,
            particle as u32,
            [position[0] + 8.0, position[1] + 2.0, position[2] - 3.0],
        );
    }
    settle(&mut world, 1);
    let after = world.inspect_soft_particles(handle);
    for (particle, (moved, was)) in after.iter().zip(&before).enumerate() {
        let drift = distance([moved[0] - 8.0, moved[1] - 2.0, moved[2] + 3.0], *was);
        assert!(
            drift < 0.05,
            "particle {particle} must continue from where it was placed, drifted {drift}"
        );
    }
}

#[test]
fn a_soft_body_carries_the_velocity_it_was_given() {
    let mut world = observed_world(static_config());
    let handle = world.add_soft_body(
        SoftBodyDesc::new(vec![[0.0, 0.0, 0.0]], Vec::new())
            .radius(0.1)
            .position([0.0, 5.0, 0.0]),
    );
    world.set_soft_particle_velocity(handle, 0, [3.0, 0.0, 1.0]);
    let before = world.inspect_soft_particles(handle)[0];
    settle(&mut world, 10);
    let after = world.inspect_soft_particles(handle)[0];
    let expected = 10.0 * DT;
    assert!(
        (after[0] - before[0] - 3.0 * expected).abs() < 1e-3
            && (after[2] - before[2] - expected).abs() < 1e-3,
        "a placed velocity must carry the particle, got {after:?} from {before:?}"
    );
}

#[test]
fn an_attached_particle_follows_the_anchor_it_was_attached_to() {
    let mut world = observed_world(static_config());
    let carrier = anchor(&mut world, [0.0, 5.0, 0.0]);
    let handle = world.add_soft_body(chain([0.0, 5.0, 0.0]));
    settle(&mut world, 30);
    world.attach_soft_particle(handle, 0, carrier, [0.0; 3]);
    settle(&mut world, 60);
    assert!(
        distance(world.inspect_soft_particles(handle)[0], [0.0, 5.0, 0.0]) < 0.05,
        "an attached particle must hold its anchor, got {:?}",
        world.inspect_soft_particles(handle)[0]
    );
    world.set_position(carrier, [6.0, 5.0, 0.0]);
    settle(&mut world, 120);
    let carried = world.inspect_soft_particles(handle)[0];
    assert!(
        distance(carried, [6.0, 5.0, 0.0]) < 0.2,
        "an attached particle must be carried to its moved anchor, got {carried:?}"
    );
}

#[test]
fn a_detached_particle_leaves_the_anchor_behind() {
    let mut world = observed_world(static_config());
    let carrier = anchor(&mut world, [0.0, 5.0, 0.0]);
    let handle = world.add_soft_body(chain([0.0, 5.0, 0.0]));
    settle(&mut world, 30);
    world.attach_soft_particle(handle, 0, carrier, [0.0; 3]);
    settle(&mut world, 60);
    world.detach_soft_particle(handle, 0);
    let position = world.inspect_soft_particles(handle)[0];
    world.set_position(carrier, [6.0, 5.0, 0.0]);
    settle(&mut world, 120);
    let detached = world.inspect_soft_particles(handle)[0];
    assert!(
        distance(detached, position) < 0.05,
        "a detached particle must stay where it was released, drifted to {detached:?} from {position:?}"
    );
}

#[test]
fn a_particle_without_an_attachment_refuses_detachment() {
    let mut world = observed_world(gravity_config());
    let handle = world.add_soft_body(chain([0.0, 5.0, 0.0]));
    settle(&mut world, 4);
    let anchor = anchor(&mut world, [0.0, 5.0, 0.0]);
    world.attach_soft_particle(handle, 0, anchor, [0.0; 3]);
    let detached = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.detach_soft_particle(handle, 1);
    }));
    assert!(
        detached.is_err(),
        "detaching a particle that carries no attachment must fail"
    );
}
