mod common;

use crate::common::shared;
use dynamis_gpu::{Backend, GpuContext, GpuRequest, GpuUnavailable, LimitsPolicy, PowerPreference};
use wgpu::Features;

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
    assert_eq!(request.optional_features, GpuRequest::PROFILING_FEATURES);
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
fn timestamp_period_is_reported() {
    assert!(shared().timestamp_period_ns() > 0.0);
}
