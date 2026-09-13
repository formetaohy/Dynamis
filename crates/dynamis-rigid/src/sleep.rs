use super::streams::RigidStream;
use dynamis_abi::{COUNTER_SLEPT, COUNTER_WOKE, COUNTER_WOKE_DEFERRED};
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernels::{CORE, rows};
use dynamis_pass::Resources;
use dynamis_pass::Stage;
use dynamis_state::Count;
use dynamis_state::StateStream;
use dynamis_state::StepFrame;

pub struct Sleep {
    island_aggregate: Stage,
    island_broadcast: Stage,
}

impl Sleep {
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            island_aggregate: Stage::build(
                context,
                "island_aggregate",
                rows(
                    context,
                    include_str!("../shaders/island_aggregate.wgsl"),
                    CORE,
                    Count::Dynamic.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("island_state", RigidStream::IslandState.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                ],
                &[],
            ),
            island_broadcast: Stage::build(
                context,
                "island_broadcast",
                rows(
                    context,
                    include_str!("../shaders/island_broadcast.wgsl"),
                    CORE,
                    Count::Dynamic.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("island_state", RigidStream::IslandState.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                    ("slept_count", dynamis_state::counter(COUNTER_SLEPT)),
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
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &StepFrame,
    ) {
        self.island_aggregate
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        self.island_broadcast
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
    }
}
