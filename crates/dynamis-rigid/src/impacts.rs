use super::streams::RigidStream;
use crate::CONTACT_COUNTERS;
use crate::RigidFrame;
use dynamis_abi::{COUNTER_CONTACTS, COUNTER_IMPACTS, COUNTER_REFUSED_IMPACTS};
use dynamis_gpu::Resources;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::Stage;
use dynamis_shader::stream;
use dynamis_state::StateStream;

pub struct Impacts {
    impacts: Stage,
}

impl Impacts {
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            impacts: Stage::build(
                context,
                "impacts",
                stream(
                    context,
                    include_str!("../shaders/impacts.wgsl"),
                    CONTACT_COUNTERS,
                    "work",
                    RigidStream::Contacts,
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
                    ("colliders", StateStream::Colliders.whole()),
                    ("impacts", RigidStream::Impacts.whole()),
                    ("impact_count", dynamis_state::counter(COUNTER_IMPACTS)),
                    ("spillover", dynamis_state::counter(COUNTER_REFUSED_IMPACTS)),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
        }
    }

    pub fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        if !frame.impacts {
            return;
        }
        self.impacts.record_stream(recorder, streams);
    }
}
