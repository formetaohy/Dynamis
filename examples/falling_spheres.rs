use dynamis::{BodyDesc, BodyHandle, GpuContext, PhysicsConfig, Simulation};

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

    let mut simulation = Simulation::new(gpu, BODY_COUNT + 32, PhysicsConfig::default());

    let platform = 5;
    for gx in 0..platform {
        for gz in 0..platform {
            let center = [
                (gx as f32 - (platform as f32 - 1.0) * 0.5) * 11.9,
                -GROUND_RADIUS,
                (gz as f32 - (platform as f32 - 1.0) * 0.5) * 11.9,
            ];
            simulation.spawn(
                BodyDesc::static_sphere(GROUND_RADIUS).position(center),
            );
        }
    }

    let mut balls: Vec<BodyHandle> = Vec::new();
    for index in 0..BODY_COUNT {
        let x = ((index % 16) as f32 - 7.5) * 1.25;
        let z = ((index / 16) as f32 - 3.5) * 1.25;
        let y = 6.0 + (index % 8) as f32 * 2.0;
        let radius = 0.4 + (index % 5) as f32 * 0.08;
        let ball = simulation.spawn(
            BodyDesc::sphere(radius)
                .position([x, y, z])
                .restitution(0.7),
        );
        balls.push(ball);
    }

    for frame in 0..FRAMES {
        simulation.step(DT);
        if frame < 100 || frame % 50 == 0 || frame == FRAMES - 1 {
            let mut lowest = f32::INFINITY;
            let mut highest = f32::NEG_INFINITY;
            for ball in &balls {
                let state = simulation.read_state(*ball);
                lowest = lowest.min(state.position[1]);
                highest = highest.max(state.position[1]);
            }
            println!(
                "frame {frame:>4}: bodies {}  y-range [{:.3}, {:.3}]",
                simulation.count(),
                lowest,
                highest
            );
        }
    }
    println!("completed {FRAMES} frames");
}
