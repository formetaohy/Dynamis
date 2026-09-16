use crate::streams::StateStream;
use dynamis_gpu::{ComputeRecorder, GpuContext, ResourceSource};
use dynamis_pass::{Execution, PassRuntime, Stage, domain_passes};

pub struct ConsumeStreams {
    consume: Stage,
}

impl PassRuntime<()> for ConsumeStreams {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            consume: Stage::build(
                context,
                "consume_streams",
                dynamis_shader::workgroups(
                    context,
                    include_str!("../shaders/consume_streams.wgsl"),
                    dynamis_shader::CORE,
                ),
                streams,
                &[
                    ("row_streams", StateStream::RowStreams.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        _: &(),
    ) {
        self.consume.record_workgroups(recorder, streams, 1);
    }
}

domain_passes!(
    StatePasses,
    StateRuntime,
    (),
    consume_streams: ConsumeStreams => Execution::STEP => &["commit"],
);
