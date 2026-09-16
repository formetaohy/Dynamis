use super::common::{gravity_config, observed_world, settle, static_config};
use dynamis_model::{BodyDesc, ColliderDesc, CollisionFilter, ConstraintDesc, Shape, SoftBodyDesc};

const BARRIER: CollisionFilter = CollisionFilter::new(0b0010, 0b0010);

fn net(height: f32, filter: CollisionFilter) -> SoftBodyDesc {
    let mut net = SoftBodyDesc::net(
        vec![
            [-0.5, 0.0, -0.5],
            [0.5, 0.0, -0.5],
            [0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
        ],
        vec![[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]],
    );
    net.radius = 0.1;
    net.friction = 0.5;
    net.position = [0.0, height, 0.0];
    net.filter = filter;
    net
}

fn slab(filter: CollisionFilter) -> BodyDesc {
    BodyDesc::cuboid([5.0, 0.5, 5.0])
        .mass(0.0)
        .position([0.0, -0.5, 0.0])
        .filter(filter)
}

fn mean_height(positions: &[[f32; 3]]) -> f32 {
    positions.iter().map(|position| position[1]).sum::<f32>() / positions.len() as f32
}

#[test]
fn a_soft_body_falls_through_a_body_its_filter_excludes() {
    let mut excluded = observed_world(gravity_config());
    excluded.spawn(slab(BARRIER));
    let handle = excluded.add_soft_body(net(1.0, CollisionFilter::DEFAULT));
    settle(&mut excluded, 90);
    let through = mean_height(&excluded.inspect_soft_particles(handle));

    let mut admitted = observed_world(gravity_config());
    admitted.spawn(slab(CollisionFilter::DEFAULT));
    let handle = admitted.add_soft_body(net(1.0, CollisionFilter::DEFAULT));
    settle(&mut admitted, 90);
    let resting = mean_height(&admitted.inspect_soft_particles(handle));

    assert!(
        resting > 0.0,
        "an admitted soft body must rest on the slab, got {resting}"
    );
    assert!(
        through < resting - 1.0,
        "an excluded soft body must pass the slab, got {through} against {resting}"
    );
}

#[test]
fn a_soft_body_honours_the_filter_of_the_collider_it_meets() {
    let mut excluded = observed_world(gravity_config());
    excluded.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::cuboid([5.0, 0.5, 5.0])).filter(BARRIER))
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let handle = excluded.add_soft_body(net(1.0, CollisionFilter::DEFAULT));
    settle(&mut excluded, 90);
    let through = mean_height(&excluded.inspect_soft_particles(handle));

    let mut admitted = observed_world(gravity_config());
    admitted.spawn(
        BodyDesc::new(
            ColliderDesc::new(Shape::cuboid([5.0, 0.5, 5.0])).filter(CollisionFilter::DEFAULT),
        )
        .mass(0.0)
        .position([0.0, -0.5, 0.0]),
    );
    let handle = admitted.add_soft_body(net(1.0, CollisionFilter::DEFAULT));
    settle(&mut admitted, 90);
    let resting = mean_height(&admitted.inspect_soft_particles(handle));

    assert!(
        resting > 0.0,
        "a collider filter that admits the soft body must hold it, got {resting}"
    );
    assert!(
        through < resting - 1.0,
        "a collider filter that excludes the soft body must let it pass, got {through}"
    );
}

fn cloud(position: [f32; 3], filter: CollisionFilter) -> SoftBodyDesc {
    let mut cloud = SoftBodyDesc::new(
        vec![[0.0, 0.0, 0.0], [0.0, 0.4, 0.0], [0.4, 0.0, 0.0]],
        Vec::new(),
    );
    cloud.radius = 0.25;
    cloud.position = position;
    cloud.filter = filter;
    cloud
}

fn closest(first: &[[f32; 3]], second: &[[f32; 3]]) -> f32 {
    first
        .iter()
        .flat_map(|left| {
            second.iter().map(move |right| {
                let delta = [right[0] - left[0], right[1] - left[1], right[2] - left[2]];
                (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
            })
        })
        .fold(f32::MAX, f32::min)
}

#[test]
fn soft_bodies_only_push_when_their_filters_intersect() {
    let mut excluded = observed_world(static_config());
    let first = excluded.add_soft_body(cloud([0.0, 0.0, 0.0], CollisionFilter::new(1, 1)));
    let second = excluded.add_soft_body(cloud([0.02, 0.0, 0.0], BARRIER));
    settle(&mut excluded, 20);
    let apart = closest(
        &excluded.inspect_soft_particles(first),
        &excluded.inspect_soft_particles(second),
    );

    let mut admitted = observed_world(static_config());
    let first = admitted.add_soft_body(cloud([0.0, 0.0, 0.0], CollisionFilter::new(1, 1)));
    let second = admitted.add_soft_body(cloud([0.02, 0.0, 0.0], CollisionFilter::DEFAULT));
    settle(&mut admitted, 20);
    let pushed = closest(
        &admitted.inspect_soft_particles(first),
        &admitted.inspect_soft_particles(second),
    );

    assert!(
        apart < 0.1,
        "soft bodies that exclude one another must stay overlapped, got {apart}"
    );
    assert!(
        pushed > 0.4,
        "soft bodies that admit one another must separate, got {pushed}"
    );
}

fn ccd_approach(joint: bool) -> f32 {
    let mut world = observed_world(static_config());
    let anchor = world.spawn(BodyDesc::static_sphere(0.5).position([0.0, 0.0, 0.0]));
    let bullet = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 8.0, 0.0])
            .velocity([0.0, -200.0, 0.0])
            .ccd(true),
    );
    if joint {
        world.add_constraint(
            anchor,
            bullet,
            ConstraintDesc::six_dof([0.0; 3], [0.0; 3], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        );
    }
    settle(&mut world, 4);
    world.read_state(bullet).position[1]
}

#[test]
fn ccd_retreats_only_pairs_no_joint_has_closed_to_contacts() {
    let stopped = ccd_approach(false);
    let passed = ccd_approach(true);
    assert!(
        (0.6..1.4).contains(&stopped),
        "ccd must stop a bullet at the anchor it may touch, got {stopped}"
    );
    assert!(
        passed < -1.0,
        "a joint that disables collisions must also close the pair to ccd, got {passed}"
    );
}

#[test]
fn a_runtime_filter_rewrites_the_pair_it_owns() {
    let mut world = observed_world(gravity_config());
    let handle = world.add_soft_body(net(1.0, CollisionFilter::DEFAULT));
    let ground = world.spawn(slab(CollisionFilter::DEFAULT));
    world.set_collision_filter(ground, BARRIER);
    settle(&mut world, 90);
    let height = mean_height(&world.inspect_soft_particles(handle));
    assert!(
        height < -1.0,
        "a body filter rewritten at runtime must open the pair it owns, got {height}"
    );
}
