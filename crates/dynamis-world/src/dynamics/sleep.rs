use super::Frame;
use super::stage::{CORE, Count, Coverage, Stage, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_SLEPT, COUNTER_WOKE, COUNTER_WOKE_DEFERRED};

pub(super) struct Sleep {
    island_aggregate: Stage,
    island_broadcast: Stage,
}

impl Sleep {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers) -> Self {
        Self {
            island_aggregate: Stage::build(
                context,
                "island_aggregate",
                include_str!("shaders/island_aggregate.wgsl"),
                CORE,
                Coverage::Live(Count::Dynamic),
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("island_parents", whole(&buffers.island_parents)),
                    ("island_state", whole(&buffers.island_state)),
                    ("wake_flags", whole(&buffers.wake_flags)),
                ],
                &[],
            ),
            island_broadcast: Stage::build(
                context,
                "island_broadcast",
                include_str!("shaders/island_broadcast.wgsl"),
                CORE,
                Coverage::Live(Count::Dynamic),
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("island_parents", whole(&buffers.island_parents)),
                    ("island_state", whole(&buffers.island_state)),
                    ("wake_flags", whole(&buffers.wake_flags)),
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
        buffers: &WorldBuffers,
        frame: &Frame,
    ) {
        self.island_aggregate.record(recorder, buffers, frame);
        self.island_broadcast.record(recorder, buffers, frame);
    }
}
