use super::Count;
use super::Frame;
use super::buffers::{RigidBuffers, StreamId};
use super::shader;
use super::shader::CORE;
use crate::dynamics::engine::Stage;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_SLEPT, COUNTER_WOKE, COUNTER_WOKE_DEFERRED};

pub(super) struct Sleep {
    island_aggregate: Stage,
    island_broadcast: Stage,
}

impl Sleep {
    pub(super) fn build(context: &GpuContext, buffers: &RigidBuffers) -> Self {
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
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("body_descs", StreamId::BodyDescriptors.whole()),
                    ("island_parents", StreamId::IslandParents.whole()),
                    ("island_state", StreamId::IslandState.whole()),
                    ("wake_flags", StreamId::WakeFlags.whole()),
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
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("body_descs", StreamId::BodyDescriptors.whole()),
                    ("island_parents", StreamId::IslandParents.whole()),
                    ("island_state", StreamId::IslandState.whole()),
                    ("wake_flags", StreamId::WakeFlags.whole()),
                    ("slept_count", buffers.counter(COUNTER_SLEPT)),
                    ("woke_count", buffers.counter(COUNTER_WOKE)),
                    (
                        "deferred_woke_count",
                        buffers.counter(COUNTER_WOKE_DEFERRED),
                    ),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        self.island_aggregate
            .record_rows(recorder, buffers, Count::Dynamic.rows(&frame.params));
        self.island_broadcast
            .record_rows(recorder, buffers, Count::Dynamic.rows(&frame.params));
    }
}
