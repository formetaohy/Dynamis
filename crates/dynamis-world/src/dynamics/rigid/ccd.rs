use super::Count;
use super::Frame;
use super::buffers::RigidBuffers;
use super::shader;
use super::shader::{CORE, GEOMETRY};
use crate::dynamics::engine::{Stage, whole};
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_PAIRS;

pub(super) struct Ccd {
    sweep: Stage,
    apply: Stage,
}

impl Ccd {
    pub(super) fn build(context: &GpuContext, buffers: &RigidBuffers) -> Self {
        Self {
            sweep: Stage::build(
                context,
                "ccd_sweep",
                shader::stream(
                    context,
                    include_str!("shaders/ccd_sweep.wgsl"),
                    GEOMETRY,
                    "work",
                    buffers.pair_capacity(),
                ),
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("colliders", whole(&buffers.colliders)),
                    ("pair_major", whole(&buffers.pair_major)),
                    ("pair_minor", whole(&buffers.pair_minor)),
                    ("pair_count", buffers.counter(COUNTER_PAIRS)),
                    ("collider_owners", whole(&buffers.collider_owners)),
                    ("ccd_factor", whole(&buffers.ccd_factor)),
                    ("ccd_impact", whole(&buffers.ccd_impact)),
                ],
                &buffers.shape_resources(),
            ),
            apply: Stage::build(
                context,
                "ccd_apply",
                shader::rows(
                    context,
                    include_str!("shaders/ccd_apply.wgsl"),
                    CORE,
                    Count::Dynamic,
                ),
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.body_states)),
                    ("ccd_factor", whole(&buffers.ccd_factor)),
                    ("ccd_impact", whole(&buffers.ccd_impact)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, frame: &Frame) {
        self.sweep.record_stream(recorder);
        self.apply
            .record_rows(recorder, Count::Dynamic.rows(&frame.params));
    }
}
