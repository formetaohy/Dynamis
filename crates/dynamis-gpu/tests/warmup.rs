use dynamis_gpu::{
    BindingKind, BindingSpec, ComputeProgram, ComputeRecorder, GpuContext, GpuRequest,
    WarmupBudget, WarmupProgress,
};
use std::time::Duration;

const TRIVIAL: &str = r#"
@group(0) @binding(0) var<storage, read_write> sink: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    sink[gid.x % 256u] = gid.x;
}
"#;

fn isolated() -> GpuContext {
    pollster::block_on(GpuContext::open(&GpuRequest::default())).expect("warmup test gpu")
}

fn program(label: &str) -> ComputeProgram {
    let bindings = [BindingSpec {
        binding: 0,
        kind: BindingKind::ReadWriteStorage,
    }];
    ComputeProgram::new(label, TRIVIAL, "main", &[&bindings])
}

#[test]
fn declaring_a_program_defers_compilation_until_warmup() {
    let context = isolated();
    let handle = context.declare(program("deferred"));
    assert!(!handle.is_warmed(), "declaration must not compile");
    assert!(!context.is_warm());
    let progress = context.warmup(WarmupBudget::All);
    assert_eq!(progress, WarmupProgress { ready: 1, total: 1 });
    assert!(progress.complete());
    assert!(handle.is_warmed());
    assert!(context.is_warm());
}

#[test]
fn declaring_the_same_program_shares_one_compilation() {
    let context = isolated();
    let first = context.declare(program("shared"));
    let second = context.declare(program("shared"));
    assert_eq!(context.warmup(WarmupBudget::All).total, 1);
    assert!(first.is_warmed() && second.is_warmed());
}

#[test]
fn a_zero_budget_compiles_one_pipeline_per_call() {
    let context = isolated();
    let first = context.declare(program("first"));
    let second = context.declare(program("second"));
    let progress = context.warmup(WarmupBudget::Within(Duration::ZERO));
    assert_eq!(progress, WarmupProgress { ready: 1, total: 2 });
    assert!(first.is_warmed());
    assert!(!second.is_warmed());
    assert!(
        context
            .warmup(WarmupBudget::Within(Duration::ZERO))
            .complete()
    );
    assert!(second.is_warmed());
}

#[test]
#[should_panic(expected = "must be warmed")]
fn recording_a_cold_pipeline_fails_fast() {
    let context = isolated();
    let handle = context.declare(program("cold"));
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cold"),
        });
    let mut recorder = ComputeRecorder::begin(&mut encoder, "cold", context.workgroups_per_row());
    recorder.record(handle.pipeline(), &[], 1);
}
