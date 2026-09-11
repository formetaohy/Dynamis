use dynamis::{BodyDesc, GpuContext, PhysicsConfig, Simulation, WarmupBudget};
use dynamis_example_profile::Profiler;
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
    let adapter = gpu.adapter_info();

    let mut profiler = Profiler::new();
    let mut sim = profiler.measure("simulation_new", || {
        Simulation::new(gpu.clone(), PhysicsConfig::default())
    });
    profiler.measure("warmup", || sim.warmup(WarmupBudget::All));
    profiler.measure("spawn_scene", || {
        spawn_scene(&mut sim, bodies);
        sim.spawn(
            BodyDesc::cuboid([30.0, 0.5, 30.0])
                .mass(0.0)
                .position([0.0, -0.5, 0.0]),
        );
    });
    println!(
        "benchmark: {} bodies, {} steps on {} {:?} backend {:?}",
        sim.count(),
        steps,
        adapter.name,
        adapter.device_type,
        adapter.backend,
    );

    let begin = Instant::now();
    for _ in 0..WARMUP_STEPS {
        profiler.measure("step", || sim.step(1.0 / 60.0));
    }
    profiler.measure("wait", || sim.wait());

    let mut gpu_frames = 0u64;
    for _ in 0..steps {
        profiler.measure("step", || sim.step(1.0 / 60.0));
        gpu_frames += record_passes(&mut profiler, &sim);
    }
    profiler.measure("wait", || sim.wait());
    gpu_frames += record_passes(&mut profiler, &sim);
    let total_wall = begin.elapsed();

    let step_phase = profiler.phase("step").expect("step samples exist");
    let wait_phase = profiler.phase("wait").expect("wait samples exist");
    let step_summary = step_phase.summary();
    let wait_summary = wait_phase.summary();

    println!("{}", profiler.render());
    println!("cpu  submit mean   : {}", format_elapsed(step_summary.mean));
    println!("cpu  submit p95    : {}", format_elapsed(step_summary.p95));
    println!("gpu  drain min     : {}", format_elapsed(wait_summary.min));
    println!("gpu  drain max     : {}", format_elapsed(wait_summary.max));
    if gpu_frames == 0 {
        println!("gpu  per pass      : unavailable, this device has no timestamp queries");
    } else {
        println!(
            "gpu  per step mean : {}",
            format_elapsed(
                profiler
                    .phase("gpu total")
                    .expect("gpu totals exist")
                    .summary()
                    .mean
            )
        );
        println!("                   (resolved from hardware timestamps over {gpu_frames} frames)");
    }
    println!(
        "throughput  : {:.0} body-steps/s  simulated {:.1}s in {:.3}s",
        sim.count() as f64 * steps as f64 / total_wall.as_secs_f64(),
        steps as f64 / 60.0,
        total_wall.as_secs_f64(),
    );
}

fn record_passes(profiler: &mut Profiler, sim: &Simulation) -> u64 {
    let timings = sim.gpu_pass_timings();
    if timings.is_empty() {
        return 0;
    }
    let mut total = 0.0;
    for timing in timings {
        profiler.record(timing.label, timing.nanoseconds);
        total += timing.nanoseconds;
    }
    profiler.record("gpu total", total);
    1
}

fn format_elapsed(ns: f64) -> String {
    dynamis_example_profile::format_ns(ns)
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
