use dynamis::{
    GpuContext, Mass, PhysicsConfig, Restitution, Simulation, SphereCollider, Transform, Velocity,
    World,
};

const FRAMES: u32 = 180;
const BODY_COUNT: usize = 64;
const DT: f32 = 1.0 / 60.0;
const GROUND_RADIUS: f32 = 60.0;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let gpu = GpuContext::new().await;
    println!("adapter: {}", gpu.adapter_info().name);

    let mut world = World::new();
    let mut simulation = Simulation::new(gpu, BODY_COUNT + 32, PhysicsConfig::default());

    let platform = 5;
    for gx in 0..platform {
        for gz in 0..platform {
            let center = [
                (gx as f32 - (platform as f32 - 1.0) * 0.5) * 11.9,
                -GROUND_RADIUS,
                (gz as f32 - (platform as f32 - 1.0) * 0.5) * 11.9,
            ];
            let segment = world.spawn((
                Transform::at(center),
                Velocity::default(),
                Mass::static_body(),
                Restitution(0.4),
                SphereCollider::new(GROUND_RADIUS),
            ));
            simulation.add(segment);
        }
    }

    for index in 0..BODY_COUNT {
        let x = ((index % 16) as f32 - 7.5) * 1.25;
        let z = ((index / 16) as f32 - 3.5) * 1.25;
        let y = 6.0 + (index % 8) as f32 * 2.0;
        let entity = world.spawn((
            Transform::at([x, y, z]),
            Velocity::default(),
            Mass::new(1.0),
            Restitution(0.7),
            SphereCollider::new(0.4 + (index % 5) as f32 * 0.08),
        ));
        simulation.add(entity);
    }

    for frame in 0..FRAMES {
        simulation.step(&world, DT);
        simulation.sync_back(&mut world);
        if frame < 100 || frame % 50 == 0 || frame == FRAMES - 1 {
            let mut lowest = f32::INFINITY;
            let mut highest = f32::NEG_INFINITY;
            world
                .query::<(&Mass, &Transform)>()
                .for_each(|(mass, transform)| {
                    if !mass.is_static() {
                        lowest = lowest.min(transform.position[1]);
                        highest = highest.max(transform.position[1]);
                    }
                });
            println!(
                "frame {frame:>4}: bodies {}  y-range [{:.3}, {:.3}]",
                simulation.bodies().len(),
                lowest,
                highest
            );
        }
    }
    println!("completed {FRAMES} frames");
}
