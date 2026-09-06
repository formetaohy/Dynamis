use dynamis::{BodyDesc, BodyHandle, ConstraintDesc, GpuContext, PhysicsConfig, Simulation};

const FRAMES: u32 = 240;
const BODY_COUNT: usize = 64;
const DT: f32 = 1.0 / 60.0;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let gpu = GpuContext::new().await;
    println!("adapter: {}", gpu.adapter_info().name);

    let mut simulation = Simulation::new(gpu, 140, PhysicsConfig::default());

    simulation.spawn(
        BodyDesc::cuboid([30.0, 1.0, 30.0])
            .mass(0.0)
            .position([0.0, -1.0, 0.0]),
    );

    let mut balls: Vec<BodyHandle> = Vec::new();
    for index in 0..BODY_COUNT {
        let x = ((index % 16) as f32 - 7.5) * 1.25;
        let z = ((index / 16) as f32 - 3.5) * 1.25;
        let y = 6.0 + (index % 8) as f32 * 2.0;
        let radius = 0.4 + (index % 5) as f32 * 0.08;
        let shape = match index % 3 {
            0 => BodyDesc::sphere(radius),
            1 => BodyDesc::cuboid([radius, radius, radius]),
            _ => BodyDesc::capsule(radius, radius),
        };
        let ball = simulation
            .spawn(shape.position([x, y, z]).restitution(0.3).friction(0.8));
        balls.push(ball);
    }

    let anchor = simulation.spawn(BodyDesc::static_sphere(0.3).position([0.0, 40.0, 8.0]));
    let mut chain = Vec::new();
    for index in 0..8 {
        let link = simulation.spawn(
            BodyDesc::sphere(0.3)
                .position([0.0, 38.0 - index as f32 * 1.6, 8.0])
                .friction(0.2),
        );
        let previous = if index == 0 { anchor } else { chain[index - 1] };
        simulation.add_constraint(
            previous,
            link,
            ConstraintDesc::ball([0.0; 3], [0.0; 3]),
        );
        chain.push(link);
    }

    for frame in 0..FRAMES {
        simulation.step(DT);
        if frame < 100 || frame % 50 == 0 || frame == FRAMES - 1 {
            simulation.wait();
            let mut lowest = f32::INFINITY;
            let mut highest = f32::NEG_INFINITY;
            let mut fastest_spin = 0.0f32;
            for ball in &balls {
                let state = simulation.read_state(*ball);
                lowest = lowest.min(state.position[1]);
                highest = highest.max(state.position[1]);
                fastest_spin = fastest_spin
                    .max(state.angular_velocity[0] * state.angular_velocity[0])
                    .max(state.angular_velocity[1] * state.angular_velocity[1])
                    .max(state.angular_velocity[2] * state.angular_velocity[2]);
            }
            println!(
                "frame {frame:>4}: bodies {}  constraints {}  y-range [{:.3}, {:.3}]  top spin {:.1} rad/s",
                simulation.count(),
                simulation.constraints().len(),
                lowest,
                highest,
                fastest_spin.sqrt()
            );
        }
    }
    println!("completed {FRAMES} frames");
}
