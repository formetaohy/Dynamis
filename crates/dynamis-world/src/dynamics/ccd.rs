use super::Frame;
use super::engine::{self, Engine, domain_passes};
use super::rigid::Count;
use super::rigid::RigidStream;
use super::scene::SceneStream;
use super::streams::Streams;

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
    pub(crate) fn new(context: &GpuContext, streams: &Streams, passes: CcdPasses) -> Self {
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
                    RigidStream::PairMajor,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("colliders", SceneStream::Colliders.whole()),
                    ("pair_major", RigidStream::PairMajor.whole()),
                    ("pair_minor", RigidStream::PairMinor.whole()),
                    ("pair_count", streams.scene.counter(COUNTER_PAIRS)),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
                    ("ccd_factor", RigidStream::CcdFactor.whole()),
                    ("ccd_impact", RigidStream::CcdImpact.whole()),
                ],
                &streams.scene.shape_resources(),
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
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("ccd_factor", RigidStream::CcdFactor.whole()),
                    ("ccd_impact", RigidStream::CcdImpact.whole()),
                ],
                &[],
            ),
        }
    }

    pub(crate) fn encode(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &Streams,
        frame: &Frame,
    ) {
        let mut sweep = engine.open(encoder, self.passes.sweep);
        self.sweep.record_stream(&mut sweep, streams);
        drop(sweep);

        let mut apply = engine.open(encoder, self.passes.apply);
        self.apply
            .record_rows(&mut apply, streams, Count::Dynamic.rows(&frame.params));
        drop(apply);
    }
}
