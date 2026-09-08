use dynamis_gpu::{
    Backend, BindingKind, BindingSpec, ComputeRecorder, GpuContext, GpuPassTiming, GpuRequest,
    GpuTimer, GpuUnavailable, LimitsPolicy, PowerPreference,
};
use std::sync::OnceLock;
use wgpu::{BindGroupEntry, BufferUsages, Features, PollType};

static CONTEXT: OnceLock<GpuContext> = OnceLock::new();

fn shared() -> &'static GpuContext {
    CONTEXT.get_or_init(|| pollster::block_on(GpuContext::new()))
}

/// Enough integer churn that even a software rasteriser measures a nonzero span.
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
fn default_request_lands_on_a_native_backend() {
    assert!(matches!(
        shared().adapter_info().backend,
        Backend::Dx12 | Backend::Metal | Backend::Vulkan
    ));
}

#[test]
fn default_request_targets_only_native_backends() {
    let request = GpuRequest::default();
    assert_eq!(request.backends, GpuRequest::NATIVE_BACKENDS);
    assert_eq!(request.limits, LimitsPolicy::Adapter);
    assert_eq!(request.power_preference, PowerPreference::HighPerformance);
}

#[test]
fn available_adapters_are_all_native() {
    let adapters = pollster::block_on(GpuContext::available_adapters(
        GpuRequest::default().backends,
    ));
    assert!(!adapters.is_empty(), "this machine must expose one");
    for adapter in &adapters {
        assert!(
            matches!(
                adapter.backend,
                Backend::Dx12 | Backend::Metal | Backend::Vulkan
            ),
            "legacy backend leaked into enumeration: {}",
            adapter.name
        );
    }
}

#[test]
fn unknown_device_name_lists_what_exists() {
    let request = GpuRequest::adapter_named("no-such-adapter-anywhere");
    let error = pollster::block_on(GpuContext::open(&request))
        .err()
        .expect("an impossible adapter name must fail");
    let GpuUnavailable::DeviceNotFound {
        requested,
        available,
    } = error
    else {
        panic!("expected DeviceNotFound, got {error:?}");
    };
    assert_eq!(requested, "no-such-adapter-anywhere");
    assert!(
        !available.is_empty(),
        "the rejection must name real adapters"
    );
}

#[test]
fn missing_required_features_are_rejected() {
    let available = shared().features();
    let candidates = [
        Features::EXPERIMENTAL_RAY_TRACING_PIPELINES,
        Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
        Features::EXPERIMENTAL_MESH_SHADER,
        Features::TEXTURE_ATOMIC,
        Features::SUBGROUP_VERTEX,
        Features::MULTIVIEW,
    ];
    let unavailable = candidates
        .iter()
        .copied()
        .find(|feature| !available.contains(*feature))
        .expect("this adapter offers every experimental feature, so no rejection case exists");
    let request = GpuRequest {
        required_features: unavailable,
        ..GpuRequest::default()
    };
    let error = pollster::block_on(GpuContext::open(&request))
        .err()
        .expect("a feature the adapter lacks must be rejected, not silently dropped");
    let GpuUnavailable::MissingFeatures { missing, .. } = error else {
        panic!("expected MissingFeatures, got {error:?}");
    };
    assert!(missing.contains(unavailable));
}

#[test]
fn rejection_messages_name_what_was_asked_and_what_exists() {
    let message = GpuUnavailable::DeviceNotFound {
        requested: "rtx-9".into(),
        available: vec!["Adapter A".into(), "Adapter B".into()],
    }
    .to_string();
    assert!(message.contains("rtx-9"), "{message}");
    assert!(message.contains("Adapter B"), "{message}");
}

#[test]
fn minimum_limits_policy_is_satisfiable_and_exact() {
    let context = pollster::block_on(GpuContext::open(&GpuRequest::minimum_limits()))
        .expect("the engine's own limits must be satisfiable on this machine");
    assert_eq!(context.limits(), &GpuContext::MINIMUM_LIMITS);
    assert_eq!(
        context.limits().max_storage_buffers_per_shader_stage,
        16,
        "the widest solver bind group must stay bindable"
    );
}

#[test]
fn adapter_policy_reports_at_least_the_engine_minimum() {
    let limits = shared().limits();
    assert!(
        GpuContext::MINIMUM_LIMITS.check_limits(limits),
        "the selected adapter must clear the engine floor"
    );
}

#[test]
fn a_fresh_device_is_not_lost() {
    assert!(shared().device_lost().is_none());
    shared().assert_alive();
}

#[test]
fn timing_features_are_negotiated_when_the_adapter_offers_them() {
    let available = shared().adapter_info();
    let supports = shared().supports(GpuRequest::TIMING_FEATURES);
    assert_eq!(
        supports,
        shared().supports_pass_timing(),
        "capability reporting must be self-consistent"
    );
    assert!(
        supports,
        "{available:?} is expected to expose timestamp queries"
    );
}

#[test]
fn timestamp_period_is_reported() {
    assert!(shared().timestamp_period_ns() > 0.0);
}

#[test]
fn pass_timings_split_a_step_by_stage() {
    if !shared().supports_pass_timing() {
        return;
    }
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
            let mut idle =
                ComputeRecorder::begin_timed(&mut encoder, labels[0], Some(timer.writes(0)));
            idle.record(&pipeline, &[], 0);
        }
        {
            let mut busy =
                ComputeRecorder::begin_timed(&mut encoder, labels[1], Some(timer.writes(1)));
            busy.record(&pipeline, &[&group], 4096);
        }
        if let Some((_, timings)) = timer.capture(context.device(), &mut encoder, frame) {
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
