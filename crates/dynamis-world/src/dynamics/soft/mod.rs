mod capacity;
mod streams;

pub(crate) use capacity::{floor, plan};
pub(crate) use streams::{DOMAIN, SoftDemand, SoftStream, SoftStreams};

use crate::dynamics::Frame;
use crate::dynamics::engine::{Engine, Stage, domain_passes};
use crate::dynamics::rigid::{Count, RigidStream};
use crate::dynamics::scene::SceneStream;
use crate::dynamics::shader;
use crate::dynamics::shader::GEOMETRY_INDEX;
use crate::dynamics::streams::Streams;
use dynamis_gpu::GpuContext;

domain_passes!(
    SoftPasses,
    "soft",
    integrate => "soft_integrate",
    links => "soft_links",
    gather => "soft_gather",
    collide => "soft_collide",
    apply => "soft_apply",
);

pub(crate) struct Soft {
    passes: SoftPasses,
    integrate: Stage,
    links: Stage,
    gather: Stage,
    collide: Stage,
    apply: Stage,
}

impl Soft {
    pub(crate) fn new(context: &GpuContext, streams: &Streams, passes: SoftPasses) -> Self {
        let particles = SoftStream::Particles;
        Self {
            passes,
            integrate: Stage::build(
                context,
                "soft_integrate",
                shader::stream(
                    context,
                    include_str!("shaders/soft_integrate.wgsl"),
                    shader::CORE,
                    "work",
                    particles,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("particles", particles.whole()),
                ],
                &[],
            ),
            links: Stage::build(
                context,
                "soft_links",
                shader::stream(
                    context,
                    include_str!("shaders/soft_links.wgsl"),
                    shader::CORE,
                    "work",
                    SoftStream::Links,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("links", SoftStream::Links.whole()),
                    ("link_deltas", SoftStream::LinkDeltas.whole()),
                ],
                &[],
            ),
            gather: Stage::build(
                context,
                "soft_gather",
                shader::stream(
                    context,
                    include_str!("shaders/soft_gather.wgsl"),
                    shader::CORE,
                    "work",
                    particles,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("links", SoftStream::Links.whole()),
                    ("link_deltas", SoftStream::LinkDeltas.whole()),
                    ("adjacency", SoftStream::Adjacency.whole()),
                ],
                &[],
            ),
            collide: Stage::build(
                context,
                "soft_collide",
                shader::stream(
                    context,
                    include_str!("shaders/soft_collide.wgsl"),
                    GEOMETRY_INDEX,
                    "work",
                    particles,
                ),
                streams,
                &[
                    ("entry_keys", RigidStream::GridEntryKeys.whole()),
                    ("entry_colliders", RigidStream::GridEntryColliders.whole()),
                    ("counters", SceneStream::Counters.whole()),
                    ("params", SceneStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("colliders", SceneStream::Colliders.whole()),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
                    ("collider_aabbs", RigidStream::ColliderAabbs.whole()),
                    ("reactions", SoftStream::Reactions.whole()),
                ],
                &streams.scene.shape_resources(),
            ),
            apply: Stage::build(
                context,
                "soft_apply",
                shader::rows(
                    context,
                    include_str!("shaders/soft_apply.wgsl"),
                    shader::CORE,
                    Count::Bodies,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("reactions", SoftStream::Reactions.whole()),
                    ("wake_flags", RigidStream::WakeFlags.whole()),
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
        let mut integrate = engine.open(encoder, self.passes.integrate);
        self.integrate.record_stream(&mut integrate, streams);
        drop(integrate);

        for _ in 0..frame.params.soft_iterations {
            let mut links = engine.open(encoder, self.passes.links);
            self.links.record_stream(&mut links, streams);
            drop(links);

            let mut gather = engine.open(encoder, self.passes.gather);
            self.gather.record_stream(&mut gather, streams);
            drop(gather);
        }

        let mut collide = engine.open(encoder, self.passes.collide);
        self.collide.record_stream(&mut collide, streams);
        drop(collide);

        let mut apply = engine.open(encoder, self.passes.apply);
        self.apply
            .record_rows(&mut apply, streams, Count::Bodies.rows(&frame.params));
        drop(apply);
    }
}
