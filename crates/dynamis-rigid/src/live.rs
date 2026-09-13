use super::streams::RigidStream;
use crate::RigidFrame;
use dynamis_abi::{COUNTER_LIVE, COUNTER_SPILLOVER_LIVE, Count};
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernel::{CORE, rows};
use dynamis_pass::{Resources, Stage};
use dynamis_state::StateStream;

pub struct Live {
    gather: Stage,
}

impl Live {
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            gather: Stage::build(
                context,
                "live gather",
                rows(
                    context,
                    include_str!("../shaders/live.wgsl"),
                    CORE,
                    Count::Dynamic.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("live_bodies", RigidStream::LiveBodies.whole()),
                    ("live_count", dynamis_state::counter(COUNTER_LIVE)),
                    ("spillover", dynamis_state::counter(COUNTER_SPILLOVER_LIVE)),
                ],
                &[],
            ),
        }
    }

    pub fn record(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.gather
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
    }
}
