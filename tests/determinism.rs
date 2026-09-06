use dynamis::{
    BodyDesc, ColliderDesc, ConstraintDesc, GpuContext, PhysicsConfig, Shape, Simulation,
};
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

const DT: f32 = 1.0 / 60.0;

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn serialized_gpu() -> (MutexGuard<'static, ()>, GpuContext) {
    let guard = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let context = pollster::block_on(GpuContext::new());
    (guard, context)
}

struct Scenario {
    handles: Vec<dynamis::BodyHandle>,
}

fn build_scenario(sim: &mut Simulation) -> Scenario {
    let vertices = vec![[-30.0f32, 0.0, -30.0], [30.0, 0.0, -30.0], [30.0, 0.0, 30.0], [-30.0, 0.0, 30.0]];
    let triangles = vec![[0u32, 2, 1], [0, 3, 2]];
    let floor = sim.add_mesh(&vertices, &triangles);
    sim.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(floor))).mass(0.0));
    let anchor = sim.spawn(BodyDesc::static_sphere(0.3).position([0.0, 6.0, 0.0]));
    let mut handles = vec![anchor];
    for index in 0..12 {
        let x = ((index % 4) as f32 - 1.5) * 0.5;
        let z = ((index / 4) as f32 - 0.5) * 0.5;
        let shape = match index % 3 {
            0 => Shape::sphere(0.25),
            1 => Shape::cuboid([0.25, 0.25, 0.25]),
            _ => Shape::cylinder(0.25, 0.25),
        };
        let handle = sim.spawn(
            BodyDesc::new(ColliderDesc::new(shape))
                .position([x, 4.0 + z, z])
                .restitution(0.2)
                .friction(0.5),
        );
        let previous = if index == 0 { anchor } else { handles[index] };
        sim.add_constraint(previous, handle, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
        handles.push(handle);
    }
    Scenario { handles }
}

fn snapshot(sim: &Simulation, scenario: &Scenario) -> Vec<u32> {
    scenario
        .handles
        .iter()
        .flat_map(|handle| {
            let body = sim.read_state(*handle);
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
fn solver_is_bit_deterministic_across_runs() {
    let (_guard, gpu) = serialized_gpu();
    let mut first = Simulation::new(gpu.clone(), 32, PhysicsConfig::default());
    let first_scenario = build_scenario(&mut first);
    let mut second = Simulation::new(gpu.clone(), 32, PhysicsConfig::default());
    let second_scenario = build_scenario(&mut second);
    let mut seen = HashMap::new();
    for frame in 0usize..120 {
        first.step(DT);
        second.step(DT);
        if frame % 20 == 0 || frame == 119 {
            first.wait();
            second.wait();
            let a = snapshot(&first, &first_scenario);
            let b = snapshot(&second, &second_scenario);
            assert_eq!(
                a, b,
                "determinism diverged at frame {frame}: {:?} vs {:?}",
                a, b
            );
            seen.insert(frame, a.clone());
        }
    }
    first.wait();
    second.wait();
    let a = snapshot(&first, &first_scenario);
    let b = snapshot(&second, &second_scenario);
    assert_eq!(a, b, "final states must match bit-exactly");
    let _ = first_scenario;
    let _ = second_scenario;
}
