use crate::RigidFrame;
use dynamis_abi::Count;
use dynamis_abi::{COUNTER_WOKE, COUNTER_WOKE_DEFERRED};
use dynamis_gpu::Resources;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::Stage;
use dynamis_shader::{CORE, rows};
use dynamis_state::StateStream;

pub struct Reactions {
    apply: Stage,
}

impl Reactions {
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            apply: Stage::build(
                context,
                "body_reactions",
                rows(
                    context,
                    include_str!("../shaders/body_reactions.wgsl"),
                    CORE,
                    Count::Bodies.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("reactions", StateStream::BodyReactions.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                    ("woke_count", dynamis_state::counter(COUNTER_WOKE)),
                    (
                        "deferred_woke_count",
                        dynamis_state::counter(COUNTER_WOKE_DEFERRED),
                    ),
                ],
                &[],
            ),
        }
    }

    pub fn record(
        &mut self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.apply
            .record_rows(recorder, streams, Count::Bodies.rows(&frame.params));
    }
}
