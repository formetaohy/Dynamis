use crate::RigidStream;
use dynamis_broadphase::BroadphaseStream;
use dynamis_engine::Resources;
use dynamis_engine::{Engine, Stage, domain_passes};
use dynamis_scene::{Count, Frame, SceneStream};

use dynamis_abi::COUNTER_PAIRS;
use dynamis_gpu::GpuContext;
use dynamis_kernels::{CORE, GEOMETRY, rows, stream};

domain_passes!(
    CcdPasses,
    "ccd",
    sweep => "ccd_sweep",
    apply => "ccd_apply",
);

pub struct Ccd {
    passes: CcdPasses,
    sweep: Stage,
    apply: Stage,
}

impl Ccd {
    pub fn new(context: &GpuContext, streams: &impl Resources, passes: CcdPasses) -> Self {
        Self {
            passes,
            sweep: Stage::build(
                context,
                "ccd_sweep",
                stream(
                    context,
                    include_str!("../shaders/ccd_sweep.wgsl"),
                    GEOMETRY,
                    "work",
                    BroadphaseStream::PairMajor,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("colliders", SceneStream::Colliders.whole()),
                    ("pair_major", BroadphaseStream::PairMajor.whole()),
                    ("pair_minor", BroadphaseStream::PairMinor.whole()),
                    ("pair_count", dynamis_scene::counter(COUNTER_PAIRS)),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
                    ("ccd_factor", RigidStream::CcdFactor.whole()),
                    ("ccd_impact", RigidStream::CcdImpact.whole()),
                ],
                &dynamis_scene::shape_resources(),
            ),
            apply: Stage::build(
                context,
                "ccd_apply",
                rows(
                    context,
                    include_str!("../shaders/ccd_apply.wgsl"),
                    CORE,
                    Count::Dynamic.field(),
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

    pub fn encode(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
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
