use super::Count;
use super::Frame;
use super::buffers::{RigidBuffers, StreamId};
use super::shader;
use super::shader::{CORE, GEOMETRY};
use crate::dynamics::engine::Stage;
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
                    StreamId::PairMajor,
                ),
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("body_descs", StreamId::BodyDescriptors.whole()),
                    ("colliders", StreamId::Colliders.whole()),
                    ("pair_major", StreamId::PairMajor.whole()),
                    ("pair_minor", StreamId::PairMinor.whole()),
                    ("pair_count", buffers.counter(COUNTER_PAIRS)),
                    ("collider_owners", StreamId::ColliderOwners.whole()),
                    ("ccd_factor", StreamId::CcdFactor.whole()),
                    ("ccd_impact", StreamId::CcdImpact.whole()),
                ],
                &RigidBuffers::shape_resources(),
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
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("ccd_factor", StreamId::CcdFactor.whole()),
                    ("ccd_impact", StreamId::CcdImpact.whole()),
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
        self.sweep.record_stream(recorder, buffers);
        self.apply
            .record_rows(recorder, buffers, Count::Dynamic.rows(&frame.params));
    }
}
