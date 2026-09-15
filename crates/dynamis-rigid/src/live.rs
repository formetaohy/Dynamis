use super::streams::RigidStream;
use crate::RigidFrame;
use dynamis_abi::{COUNTER_LIVE, COUNTER_LIVE_FAULTS, Count};
use dynamis_gpu::Resources;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{PassRuntime, Stage};
use dynamis_shader::{CORE, rows};
use dynamis_state::StateStream;

pub struct Live {
    gather: Stage,
}

impl PassRuntime<RigidFrame> for Live {
    fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            gather: Stage::build(
                context,
                "live gather",
                rows(
                    context,
                    include_str!("../shaders/live.wgsl"),
                    CORE,
                    Count::Dynamic.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("live_bodies", RigidStream::LiveBodies.whole()),
                    ("live_count", dynamis_state::counter(COUNTER_LIVE)),
                    ("spillover", dynamis_state::counter(COUNTER_LIVE_FAULTS)),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.gather.record_rows(
            recorder,
            streams,
            Count::Dynamic.rows(&frame.params, &frame.rows),
        );
    }
}
