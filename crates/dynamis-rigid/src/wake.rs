use crate::RigidFrame;
use dynamis_abi::Count;
use dynamis_gpu::ResourceSource;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{Execution, PassRuntime, Stage};
use dynamis_shader::{CORE, rows};
use dynamis_state::StateStream;

pub const WAKE_ALL_GATE: u32 = 2;

pub const WAKE_ALL_EXECUTION: Execution = Execution::STEP
    .and(Execution::AWAKE)
    .and(Execution::gate(WAKE_ALL_GATE));

pub struct WakeAll {
    flags: Stage,
}

impl PassRuntime<RigidFrame> for WakeAll {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            flags: Stage::build(
                context,
                "wake_all",
                rows(
                    context,
                    include_str!("../shaders/wake_all.wgsl"),
                    CORE,
                    Count::Dynamic.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        self.flags.record_rows(
            recorder,
            streams,
            Count::Dynamic.rows(&frame.params, &frame.rows),
        );
    }
}
