use crate::RigidFrame;
use dynamis_abi::Count;
use dynamis_abi::{COUNTER_WOKE, COUNTER_WOKE_DEFERRED};
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::ResourceSource;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{PassRuntime, Stage};
use dynamis_shader::{CORE, rows};
use dynamis_state::StateStream;

pub struct ApplyReactions {
    apply: Stage,
}

impl PassRuntime<RigidFrame> for ApplyReactions {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            apply: Stage::build(
                context,
                "body_reactions",
                rows(
                    context,
                    include_str!("../shaders/body_reactions.wgsl"),
                    CORE,
                    Count::Bodies.bound(),
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
                    ("body_admitted", BroadphaseStream::BodyAdmitted.whole()),
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
        self.apply.record_rows(
            recorder,
            streams,
            Count::Bodies.rows(&frame.params, &frame.rows),
        );
    }
}
