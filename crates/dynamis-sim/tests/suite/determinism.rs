use super::common::{DT, sim};
use dynamis_model::{BodyDesc, BodyHandle, ColliderDesc, ConstraintDesc, PhysicsConfig, Shape};
use dynamis_sim::Simulation;

struct Scenario {
    handles: Vec<BodyHandle>,
}

fn build_scenario(world: &mut Simulation) -> Scenario {
    let vertices = vec![
        [-30.0f32, 0.0, -30.0],
        [30.0, 0.0, -30.0],
        [30.0, 0.0, 30.0],
        [-30.0, 0.0, 30.0],
    ];
    let triangles = vec![[0u32, 2, 1], [0, 3, 2]];
    let floor = world.add_mesh(&vertices, &triangles);
    world.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(floor))).mass(0.0));
    let anchor = world.spawn(BodyDesc::static_sphere(0.3).position([0.0, 6.0, 0.0]));
    let mut handles = vec![anchor];
    for index in 0..12 {
        let x = ((index % 4) as f32 - 1.5) * 0.5;
        let z = ((index / 4) as f32 - 0.5) * 0.5;
        let shape = match index % 3 {
            0 => Shape::sphere(0.25),
            1 => Shape::cuboid([0.25, 0.25, 0.25]),
            _ => Shape::cylinder(0.25, 0.25),
        };
        let handle = world.spawn(
            BodyDesc::new(ColliderDesc::new(shape))
                .position([x, 4.0 + z, z])
                .restitution(0.2)
                .friction(0.5),
        );
        let previous = if index == 0 { anchor } else { handles[index] };
        world.add_constraint(previous, handle, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
        handles.push(handle);
    }
    Scenario { handles }
}

fn snapshot(world: &Simulation, scenario: &Scenario) -> Vec<u32> {
    scenario
        .handles
        .iter()
        .flat_map(|handle| {
            let body = world.read_state(*handle);
            [
                body.position[0].to_bits(),
                body.position[1].to_bits(),
                body.position[2].to_bits(),
                body.velocity[0].to_bits(),
                body.velocity[1].to_bits(),
                body.velocity[2].to_bits(),
            ]
        })
        .collect()
}

#[test]
fn solver_is_bit_deterministic_across_worlds() {
    let mut first = sim(32, PhysicsConfig::default());
    let first_scenario = build_scenario(&mut first);
    let mut second = sim(32, PhysicsConfig::default());
    let second_scenario = build_scenario(&mut second);
    let frames = [0u64, 24, 48, 71];
    for frame in 0usize..72 {
        first.step(DT);
        second.step(DT);
        if frames.contains(&(frame as u64)) {
            first.wait();
            second.wait();
            let a = snapshot(&first, &first_scenario);
            let b = snapshot(&second, &second_scenario);
            assert_eq!(
                a, b,
                "determinism diverged at frame {frame}: {:?} vs {:?}",
                a, b
            );
        }
    }
    first.wait();
    second.wait();
    assert_eq!(
        snapshot(&first, &first_scenario),
        snapshot(&second, &second_scenario),
        "final states must match bit-exactly"
    );
}
