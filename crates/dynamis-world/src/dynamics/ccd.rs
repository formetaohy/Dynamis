use super::engine::{self, Engine, domain_passes};
use super::rigid::Count;
use super::rigid::Frame;
use super::rigid::buffers::{RigidBuffers, StreamId};
use super::rigid::shader;
use super::rigid::shader::{CORE, GEOMETRY};
use dynamis_gpu::GpuContext;
use dynamis_layout::COUNTER_PAIRS;

domain_passes!(
    CcdPasses,
    "ccd",
    sweep => "ccd_sweep",
    apply => "ccd_apply",
);

pub(crate) struct Ccd {
    passes: CcdPasses,
    sweep: engine::Stage,
    apply: engine::Stage,
}

impl Ccd {
    pub(crate) fn new(context: &GpuContext, buffers: &RigidBuffers, passes: CcdPasses) -> Self {
        Self {
            passes,
            sweep: engine::Stage::build(
                context,
                "ccd_sweep",
                shader::stream(
                    context,
                    include_str!("rigid/shaders/ccd_sweep.wgsl"),
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
            apply: engine::Stage::build(
                context,
                "ccd_apply",
                shader::rows(
                    context,
                    include_str!("rigid/shaders/ccd_apply.wgsl"),
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

    pub(crate) fn encode(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        let mut sweep = engine.open(encoder, self.passes.sweep);
        self.sweep.record_stream(&mut sweep, buffers);
        drop(sweep);

        let mut apply = engine.open(encoder, self.passes.apply);
        self.apply
            .record_rows(&mut apply, buffers, Count::Dynamic.rows(&frame.params));
        drop(apply);
    }
}
