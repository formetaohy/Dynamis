mod capacity;
mod streams;

pub use capacity::{SoftCapacity, capacity, floor, plan};
pub use streams::{DOMAIN, SoftDemand, SoftStream, SoftStreams};

use dynamis_broadphase::BroadphaseStream;
use dynamis_engine::Resources;
use dynamis_engine::{Engine, Stage, domain_passes};
use dynamis_gpu::GpuContext;
use dynamis_kernels::{CORE, GEOMETRY_INDEX, rows, stream};
use dynamis_scene::{Count, Frame, SceneStream};

const PARTICLE_SHAPE: &[&str] = &[include_str!("../shaders/particle_shape.wgsl")];

fn particle_index() -> Vec<&'static str> {
    let mut fragments = dynamis_kernels::GRID_INDEX.to_vec();
    fragments.extend_from_slice(PARTICLE_SHAPE);
    fragments
}

domain_passes!(
    SoftPasses,
    "soft",
    bounds => "soft_bounds",
    entries => "soft_entries",
    integrate => "soft_integrate",
    links => "soft_links",
    gather => "soft_gather",
    collide => "soft_collide",
    apply => "soft_apply",
);

pub struct Soft {
    passes: SoftPasses,
    bounds: Stage,
    entries: Stage,
    integrate: Stage,
    links: Stage,
    gather: Stage,
    collide: Stage,
    apply: Stage,
}

impl Soft {
    pub fn new(context: &GpuContext, streams: &impl Resources, passes: SoftPasses) -> Self {
        let particles = SoftStream::Particles;
        let index = particle_index();
        Self {
            passes,
            bounds: Stage::build(
                context,
                "soft_bounds",
                stream(
                    context,
                    include_str!("../shaders/particle_bounds.wgsl"),
                    PARTICLE_SHAPE,
                    "work",
                    particles,
                ),
                streams,
                &[
                    ("counters", SceneStream::Counters.whole()),
                    ("params", SceneStream::Params.whole()),
                    ("particles", particles.whole()),
                ],
                &[],
            ),
            entries: Stage::build(
                context,
                "soft_entries",
                stream(
                    context,
                    include_str!("../shaders/particle_entries.wgsl"),
                    &index,
                    "work",
                    particles,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", SceneStream::Counters.whole()),
                ],
                &[],
            ),
            integrate: Stage::build(
                context,
                "soft_integrate",
                stream(
                    context,
                    include_str!("../shaders/soft_integrate.wgsl"),
                    CORE,
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
                stream(
                    context,
                    include_str!("../shaders/soft_links.wgsl"),
                    CORE,
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
                stream(
                    context,
                    include_str!("../shaders/soft_gather.wgsl"),
                    CORE,
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
                stream(
                    context,
                    include_str!("../shaders/soft_collide.wgsl"),
                    GEOMETRY_INDEX,
                    "work",
                    particles,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("colliders", SceneStream::Colliders.whole()),
                    ("links", SoftStream::Links.whole()),
                    ("adjacency", SoftStream::Adjacency.whole()),
                    ("reactions", SoftStream::Reactions.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", SceneStream::Counters.whole()),
                ],
                &dynamis_scene::shape_resources(),
            ),
            apply: Stage::build(
                context,
                "soft_apply",
                rows(
                    context,
                    include_str!("../shaders/soft_apply.wgsl"),
                    CORE,
                    Count::Bodies.field(),
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("reactions", SoftStream::Reactions.whole()),
                    ("wake_flags", SceneStream::WakeFlags.whole()),
                ],
                &[],
            ),
        }
    }

    pub fn encode_bounds(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
    ) {
        let mut bounds = engine.open(encoder, self.passes.bounds);
        self.bounds.record_stream(&mut bounds, streams);
        drop(bounds);
    }

    pub fn encode_entries(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
    ) {
        let mut entries = engine.open(encoder, self.passes.entries);
        self.entries.record_stream(&mut entries, streams);
        drop(entries);
    }

    pub fn encode(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
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
