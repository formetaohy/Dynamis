use super::Count;
use super::shader;
use super::shader::CORE;
use super::streams::RigidStream;
use crate::dynamics::Frame;
use crate::dynamics::engine::Stage;
use crate::dynamics::scene::SceneStream;
use crate::dynamics::streams::Streams;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_SLEPT, COUNTER_WOKE, COUNTER_WOKE_DEFERRED};

pub(super) struct Sleep {
    island_aggregate: Stage,
    island_broadcast: Stage,
}

impl Sleep {
    pub(super) fn build(context: &GpuContext, streams: &Streams) -> Self {
        Self {
            island_aggregate: Stage::build(
                context,
                "island_aggregate",
                shader::rows(
                    context,
                    include_str!("shaders/island_aggregate.wgsl"),
                    CORE,
                    Count::Dynamic,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("island_state", RigidStream::IslandState.whole()),
                    ("wake_flags", RigidStream::WakeFlags.whole()),
                ],
                &[],
            ),
            island_broadcast: Stage::build(
                context,
                "island_broadcast",
                shader::rows(
                    context,
                    include_str!("shaders/island_broadcast.wgsl"),
                    CORE,
                    Count::Dynamic,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("island_state", RigidStream::IslandState.whole()),
                    ("wake_flags", RigidStream::WakeFlags.whole()),
                    ("slept_count", streams.scene.counter(COUNTER_SLEPT)),
                    ("woke_count", streams.scene.counter(COUNTER_WOKE)),
                    (
                        "deferred_woke_count",
                        streams.scene.counter(COUNTER_WOKE_DEFERRED),
                    ),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, streams: &Streams, frame: &Frame) {
        self.island_aggregate
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        self.island_broadcast
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
    }
}
