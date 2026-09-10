use super::FrameParams;
use super::stage::{RO, RW, Stage, UNIFORM, whole};
use crate::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_SLEPT, COUNTER_WOKE};

pub(super) struct Sleep {
    island_aggregate: Stage,
    island_broadcast: Stage,
}

impl Sleep {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            island_aggregate: Stage::build(
                context,
                "island_aggregate",
                include_str!("../shaders/island_aggregate.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RW, whole(&buffers.islands.parents)),
                    (RW, whole(&buffers.islands.state)),
                    (RW, whole(&buffers.islands.wake_flags)),
                ],
                &[],
            ),
            island_broadcast: Stage::build(
                context,
                "island_broadcast",
                include_str!("../shaders/island_broadcast.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RW, whole(&buffers.islands.parents)),
                    (RW, whole(&buffers.islands.state)),
                    (RW, whole(&buffers.islands.wake_flags)),
                    (RW, buffers.counter(COUNTER_SLEPT)),
                    (RW, buffers.counter(COUNTER_WOKE)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, params: &FrameParams) {
        self.island_aggregate.record(recorder, params.dynamic_count);
        self.island_broadcast.record(recorder, params.dynamic_count);
    }
}
