use super::stage::{GEOMETRY, Stage, shape_resources, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_PAIRS;

pub(super) struct Ccd {
    ccd_sweep: Stage,
}

impl Ccd {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            ccd_sweep: Stage::build(
                context,
                "ccd_sweep",
                include_str!("shaders/ccd_sweep.wgsl"),
                per_row,
                GEOMETRY,
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("colliders", whole(&buffers.colliders)),
                    ("pair_major", whole(&buffers.pair_major)),
                    ("pair_minor", whole(&buffers.pair_minor)),
                    ("pair_count", buffers.counter(COUNTER_PAIRS)),
                    ("collider_owners", whole(&buffers.collider_owners)),
                ],
                &shape_resources(buffers),
            ),
        }
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, buffers: &WorldBuffers) {
        self.ccd_sweep
            .record_stride(recorder, buffers.pair_capacity());
    }
}
