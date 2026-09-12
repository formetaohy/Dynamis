use super::Count;
use super::Frame;
use super::buffers::RigidBuffers;
use super::shader;
use super::shader::CORE;
use crate::dynamics::engine::{Stage, whole};
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
                shader::rows(
                    context,
                    include_str!("shaders/island_broadcast.wgsl"),
                    CORE,
                    Count::Dynamic,
                ),
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

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, frame: &Frame) {
        self.island_aggregate
            .record_rows(recorder, Count::Dynamic.rows(&frame.params));
        self.island_broadcast
            .record_rows(recorder, Count::Dynamic.rows(&frame.params));
    }
}
