use super::stage::{GEOMETRY, RO, RW, Stage, UNIFORM, shape_resources, whole};
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
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RO, whole(&buffers.bodies.colliders)),
                    (RO, whole(&buffers.contacts.pairs.major)),
                    (RO, whole(&buffers.contacts.pairs.minor)),
                    (RW, buffers.counter(COUNTER_PAIRS)),
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
