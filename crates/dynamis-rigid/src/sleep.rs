use super::streams::RigidStream;
use dynamis_abi::{COUNTER_SLEPT, COUNTER_WOKE, COUNTER_WOKE_DEFERRED};
use dynamis_engine::Resources;
use dynamis_engine::Stage;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernels::{CORE, rows};
use dynamis_scene::Count;
use dynamis_scene::Frame;
use dynamis_scene::SceneStream;

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
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("island_state", RigidStream::IslandState.whole()),
                    ("wake_flags", SceneStream::WakeFlags.whole()),
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
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("island_state", RigidStream::IslandState.whole()),
                    ("wake_flags", SceneStream::WakeFlags.whole()),
                    ("slept_count", dynamis_scene::counter(COUNTER_SLEPT)),
                    ("woke_count", dynamis_scene::counter(COUNTER_WOKE)),
                    (
                        "deferred_woke_count",
                        dynamis_scene::counter(COUNTER_WOKE_DEFERRED),
                    ),
                ],
                &[],
            ),
        }
    }

    pub fn record(&self, recorder: &mut ComputeRecorder, streams: &impl Resources, frame: &Frame) {
        self.island_aggregate
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        self.island_broadcast
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
    }
}
