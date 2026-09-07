use dynamis::{BodyDesc, GpuContext, PhysicsConfig, Simulation};
use dynamis_sim::StageId;
use dynamis_profile::{
    ProfileReport, StageSampler, StageSummary, adapter_label, summary_of,
};
use std::time::Instant;

const DEFAULT_BODIES: usize = 128;
const DEFAULT_STEPS: usize = 600;
const WARMUP_STEPS: usize = 60;

fn main() {
    let bodies = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_BODIES);
    let steps = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_STEPS);
    let gpu = pollster::block_on(GpuContext::new());
    println!(
        "benchmark: {} bodies, {} steps on {}",
        bodies,
        steps,
        adapter_label(&gpu.adapter_info())
    );
    println!("timestamps supported: {}\n", gpu.supports_timestamps());

    let mut sim = Simulation::new(gpu.clone(), bodies + 4, PhysicsConfig::default());
    spawn_scene(&mut sim, bodies);
    let ground = sim.spawn(
        BodyDesc::cuboid([30.0, 0.5, 30.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let _ = ground;
    println!("spawned {} bodies", sim.count());

    if !sim.enable_profiling() {
        println!("warning: device lacks timestamp queries, GPU profile unavailable");
    }

    for _ in 0..WARMUP_STEPS {
        sim.step(1.0 / 60.0);
    }
    sim.wait();

    let mut cpu_times = Vec::with_capacity(steps);
    let mut sampler = StageSampler::new();
    for _ in 0..steps {
        let begin = Instant::now();
        sim.step(1.0 / 60.0);
        cpu_times.push(begin.elapsed().as_nanos() as f64);
        sim.poll_stage_samples();
        if let Some(stage_samples) = sim.stage_samples() {
            sampler.record(stage_samples);
        }
    }
    sim.wait();

    let cpu = summary_of(&cpu_times);
    let gpu_total = summary_of(&sampler.total_series());
    let stages = StageId::ALL
        .iter()
        .map(|stage| {
            let series = sampler.stage_series(*stage);
            let summary = summary_of(&series);
            let share = summary.mean / gpu_total.mean.max(1e-9);
            StageSummary {
                stage: *stage,
                gpu: summary,
                share,
            }
        })
        .collect::<Vec<_>>();
    let report = ProfileReport {
        adapter: adapter_label(&gpu.adapter_info()),
        body_count: sim.count(),
        steps,
        cpu_step: cpu,
        gpu_total,
        stages,
    };
    println!("{}", report.render());
    let step_time_s = cpu.mean / 1_000_000_000.0;
    println!(
        "throughput: {:.0} body-steps/s  (simulated {:.1}s in {:.3}s)",
        sim.count() as f64 * steps as f64 / step_time_s,
        steps as f64 / 60.0,
        step_time_s * steps as f64
    );
}

fn spawn_scene(sim: &mut Simulation, count: usize) {
    for index in 0..count {
        let radius = 0.35 + (index % 5) as f32 * 0.07;
        let desc = match index % 4 {
            0 => BodyDesc::sphere(radius),
            1 => BodyDesc::cuboid([radius, radius * 0.9, radius * 1.1]),
            2 => BodyDesc::capsule(radius * 0.8, radius),
            _ => BodyDesc::cylinder(radius * 0.8, radius),
        };
        let x = (index % 16) as f32 * 1.7 - 12.75;
        let z = (index / 16) as f32 * 1.7 - 6.0;
        let y = 5.0 + (index % 13) as f32 * 1.5;
        let spin = 0.4 + (index % 7) as f32 * 0.25;
        sim.spawn(
            desc.position([x, y, z])
                .angular_velocity([spin, spin * 0.7, spin * 0.5])
                .restitution(0.2 + (index % 4) as f32 * 0.08)
                .friction(0.7),
        );
    }
}
