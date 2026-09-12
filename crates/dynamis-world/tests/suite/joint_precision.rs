use super::common::{DT, gravity_config, new_world, static_config};
use dynamis_model::{BodyDesc, BodyHandle, ConstraintDesc, DofDesc};
use dynamis_world::World;

fn anchor_of(world: &World, handle: BodyHandle, local: [f32; 3]) -> [f32; 3] {
    let state = world.read_state(handle);
    let q = state.orientation;
    let u = [q[0], q[1], q[2]];
    let s = q[3];
    let spin = [
        u[1] * local[2] - u[2] * local[1],
        u[2] * local[0] - u[0] * local[2],
        u[0] * local[1] - u[1] * local[0],
    ];
    let spin_spin = [
        u[1] * spin[2] - u[2] * spin[1],
        u[2] * spin[0] - u[0] * spin[2],
        u[0] * spin[1] - u[1] * spin[0],
    ];
    [
        state.position[0] + local[0] + 2.0 * (s * spin[0] + spin_spin[0]),
        state.position[1] + local[1] + 2.0 * (s * spin[1] + spin_spin[1]),
        state.position[2] + local[2] + 2.0 * (s * spin[2] + spin_spin[2]),
    ]
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[test]
fn ball_joint_holds_anchor_across_mass_ratio() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(
        BodyDesc::cuboid([0.2, 0.2, 0.2])
            .position([0.0, 0.2, 0.0])
            .mass(50.0),
    );
    world.add_constraint(
        first,
        second,
        ConstraintDesc::ball([0.0, 0.2, 0.0], [0.0, -0.2, 0.0]),
    );
    let mut worst = 0.0f32;
    for frame in 0..180 {
        world.apply_force(second, [0.0, 0.0, 400.0]);
        world.step(DT);
        if frame % 40 != 39 {
            continue;
        }
        world.wait();
        worst = worst.max(distance(
            anchor_of(&world, first, [0.0, 0.2, 0.0]),
            anchor_of(&world, second, [0.0, -0.2, 0.0]),
        ));
    }
    assert!(
        worst < 0.01,
        "a ball joint must hold its anchor across a 50:1 mass ratio, worst gap {worst}"
    );
}

#[test]
fn prismatic_joint_holds_lateral_axis() {
    let mut world = new_world(static_config());
    let base = world.spawn(BodyDesc::sphere(0.1).mass(0.0));
    let slider = world.spawn(BodyDesc::sphere(0.1).position([0.0, 1.0, 0.0]));
    let desc =
        ConstraintDesc::prismatic([0.0; 3], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]).limit(-10.0, 10.0);
    world.add_constraint(base, slider, desc);
    let mut worst = 0.0f32;
    for frame in 0..180 {
        world.apply_force(slider, [0.0, 5.0, 30.0]);
        world.step(DT);
        if frame % 40 != 39 {
            continue;
        }
        world.wait();
        let base_position = world.read_state(base).position;
        let slider_position = world.read_state(slider).position;
        let lateral = ((slider_position[0] - base_position[0]).powi(2)
            + (slider_position[2] - base_position[2]).powi(2))
        .sqrt();
        worst = worst.max(lateral);
    }
    assert!(
        worst < 0.01,
        "a prismatic joint must hold the slider on its axis, worst drift {worst}"
    );
}

#[test]
fn six_dof_locked_holds_pose_under_load() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.2).mass(0.0));
    let second = world.spawn(BodyDesc::cuboid([0.2, 0.2, 0.2]).position([0.5, 0.4, 0.3]));
    let desc = ConstraintDesc::six_dof([0.0; 3], [0.0; 3], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0])
        .dofs([DofDesc::locked(); 6]);
    world.add_constraint(first, second, desc);
    let mut worst = 0.0f32;
    for frame in 0..180 {
        world.apply_force(second, [0.0, 10.0, 0.0]);
        world.apply_torque(second, [50.0, 0.0, 30.0]);
        world.step(DT);
        if frame % 40 != 39 {
            continue;
        }
        world.wait();
        worst = worst.max(distance(
            world.read_state(first).position,
            world.read_state(second).position,
        ));
    }
    assert!(
        worst < 0.001,
        "a locked six-dof joint must pin the relative pose, worst gap {worst}"
    );
}

#[test]
fn hanging_chain_holds_every_anchor() {
    let mut world = new_world(gravity_config());
    let anchor = world.spawn(BodyDesc::static_sphere(0.05));
    let mut previous = anchor;
    let mut links = Vec::new();
    for index in 0..4 {
        let body = world.spawn(BodyDesc::cuboid([0.15, 0.3, 0.15]).position([
            0.0,
            -1.0 - index as f32 * 0.6,
            0.0,
        ]));
        world.add_constraint(
            previous,
            body,
            ConstraintDesc::ball([0.0, 0.3, 0.0], [0.0, -0.3, 0.0]),
        );
        previous = body;
        links.push(body);
    }
    let mut worst = 0.0f32;
    for frame in 0..200 {
        world.step(DT);
        if frame % 40 != 39 {
            continue;
        }
        world.wait();
        let mut cursor = anchor;
        for link in &links {
            worst = worst.max(distance(
                anchor_of(&world, cursor, [0.0, 0.3, 0.0]),
                anchor_of(&world, *link, [0.0, -0.3, 0.0]),
            ));
            cursor = *link;
        }
    }
    assert!(
        worst < 0.05,
        "a hanging ball chain must keep every anchor together, worst gap {worst}"
    );
}
