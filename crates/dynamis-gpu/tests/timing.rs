#![cfg(feature = "profile")]

mod common;

use crate::common::shared;
use dynamis_gpu::{BindingKind, BindingSpec, ComputeRecorder, GpuPassTiming, GpuRequest, GpuTimer};
use wgpu::{BindGroupEntry, BufferUsages, PollType};

const BURN_KERNEL: &str = r#"
@group(0) @binding(0) var<storage, read_write> sink: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    var value = gid.x;
    for (var i = 0u; i < 512u; i = i + 1u) {
        value = value * 1664525u + 1013904223u;
    }
    sink[gid.x % 4096u] = value;
}
"#;

fn burn_pipeline() -> dynamis_gpu::ComputePipeline {
    let context = shared();
    let bindings = [BindingSpec {
        binding: 0,
        kind: BindingKind::ReadWriteStorage,
    }];
    context.compute_pipeline("burn", BURN_KERNEL, "main", &[&bindings[..]], 64)
}

fn burn_group(pipeline: &dynamis_gpu::ComputePipeline) -> wgpu::BindGroup {
    let context = shared();
    let sink = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("burn sink"),
        size: 4096 * 4,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    pipeline.create_bind_group(
        context.device(),
        0,
        &[BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &sink,
                offset: 0,
                size: None,
            }),
        }],
    )
}

#[test]
fn the_default_request_negotiates_timestamp_queries() {
    assert!(
        shared().supports_pass_timing(),
        "{:?} must expose timestamp queries to a profile build",
        shared().adapter_info()
    );
    assert!(
        !GpuRequest::PROFILING_FEATURES.is_empty(),
        "a profile build must ask the device for the timing features"
    );
}

#[test]
fn pass_timings_split_a_step_by_stage() {
    let context = shared();
    let labels: &'static [&'static str] = &["idle", "busy"];
    let mut timer = GpuTimer::new(
        context.device(),
        labels,
        context.timestamp_period_ns(),
        "test",
    );
    let pipeline = burn_pipeline();
    let group = burn_group(&pipeline);
    let mut reported: Option<Vec<GpuPassTiming>> = None;
    for frame in 0..6 {
        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("timing frame"),
                });
        {
            let mut idle = ComputeRecorder::begin_timed(
                &mut encoder,
                labels[0],
                Some(timer.writes(0)),
                context.workgroups_per_row(),
            );
            idle.record(&pipeline, &[], 0);
        }
        {
            let mut busy = ComputeRecorder::begin_timed(
                &mut encoder,
                labels[1],
                Some(timer.writes(1)),
                context.workgroups_per_row(),
            );
            busy.record(&pipeline, &[&group], 4096);
        }
        if let Some((_, timings)) = timer.capture(&mut encoder, frame) {
            reported = Some(timings);
            context.queue().submit([encoder.finish()]);
            timer.arm();
            break;
        }
        context.queue().submit([encoder.finish()]);
        timer.arm();
        context
            .device()
            .poll(PollType::wait_indefinitely())
            .expect("device lost while timing");
    }
    let timings = reported.expect("a double-buffered timer must report a completed frame");
    assert_eq!(timings.len(), 2);
    for timing in &timings {
        assert!(
            timing.nanoseconds >= 0.0,
            "{} measured a negative duration",
            timing.label
        );
    }
    assert!(
        timings[1].nanoseconds > 0.0,
        "a dispatched pass must record measurable time"
    );
    assert!(
        timings[1].nanoseconds > timings[0].nanoseconds,
        "per-pass attribution must separate an empty pass ({}) from real work ({})",
        timings[0].nanoseconds,
        timings[1].nanoseconds
    );
}

#[test]
fn timing_slot_out_of_range_is_rejected() {
    let context = shared();
    let labels: &'static [&'static str] = &["only"];
    let timer = GpuTimer::new(
        context.device(),
        labels,
        context.timestamp_period_ns(),
        "bounds",
    );
    assert_eq!(timer.pass_count(), 1);
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| timer.writes(1)));
    assert!(error.is_err(), "an unknown pass slot must abort");
}

#[test]
fn a_timer_requires_at_least_one_pass() {
    let context = shared();
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        GpuTimer::new(context.device(), &[], 1.0, "empty")
    }));
    assert!(error.is_err(), "a zero-pass timer is meaningless");
}

#[test]
fn milliseconds_match_the_reported_nanoseconds() {
    let timing = GpuPassTiming {
        label: "pass",
        nanoseconds: 2_500_000.0,
    };
    assert_eq!(timing.milliseconds(), 2.5);
}
