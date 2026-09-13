mod capacity;
mod streams;

pub use capacity::{SoftCapacity, capacity, floor, plan};
pub use streams::{DOMAIN, SoftDemand, SoftStream, SoftStreams};

use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::GpuContext;
use dynamis_kernels::{CORE, GEOMETRY_INDEX, rows, stream};
use dynamis_pass::Resources;
use dynamis_pass::{Phase, Schedule, Stage, domain_passes};
use dynamis_state::{Count, StateStream, StepFrame};

const PARTICLE_SHAPE: &[&str] = &[include_str!("../shaders/particle_shape.wgsl")];

fn particle_index() -> Vec<&'static str> {
    let mut fragments = dynamis_kernels::GRID_INDEX.to_vec();
    fragments.extend_from_slice(PARTICLE_SHAPE);
    fragments
}

domain_passes!(
    SoftPasses,
    "soft",
    bounds: Phase::Prepare => "soft_bounds",
    entries: Phase::Entries => "soft_entries",
    substeps: Phase::Deform => "soft_substeps",
    collide: Phase::Deform => "soft_collide",
    apply: Phase::Deform => "soft_apply",
);

pub struct Soft {
    passes: SoftPasses,
    bounds: Stage,
    entries: Stage,
    integrate: Stage,
    reset: Stage,
    elements: Stage,
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
                    ("counters", StateStream::Counters.whole()),
                    ("params", StateStream::Params.whole()),
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
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
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
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                ],
                &[],
            ),
            reset: Stage::build(
                context,
                "soft_reset",
                stream(
                    context,
                    include_str!("../shaders/soft_reset.wgsl"),
                    CORE,
                    "work",
                    SoftStream::Elements,
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("elements", SoftStream::Elements.whole()),
                ],
                &[],
            ),
            elements: Stage::build(
                context,
                "soft_elements",
                stream(
                    context,
                    include_str!("../shaders/soft_elements.wgsl"),
                    CORE,
                    "work",
                    SoftStream::Elements,
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("elements", SoftStream::Elements.whole()),
                    ("element_deltas", SoftStream::ElementDeltas.whole()),
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
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("elements", SoftStream::Elements.whole()),
                    ("element_deltas", SoftStream::ElementDeltas.whole()),
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
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("colliders", StateStream::Colliders.whole()),
                    ("elements", SoftStream::Elements.whole()),
                    ("adjacency", SoftStream::Adjacency.whole()),
                    ("reactions", SoftStream::Reactions.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &dynamis_state::shape_resources(),
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
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("reactions", SoftStream::Reactions.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                    (
                        "woke_count",
                        dynamis_state::counter(dynamis_abi::COUNTER_WOKE),
                    ),
                    (
                        "deferred_woke_count",
                        dynamis_state::counter(dynamis_abi::COUNTER_WOKE_DEFERRED),
                    ),
                ],
                &[],
            ),
        }
    }

    pub fn record(
        &self,
        phase: Phase,
        schedule: &mut Schedule,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
        frame: &StepFrame,
    ) {
        if !frame.soft_bodies {
            return;
        }
        match phase {
            Phase::Prepare => {
                let mut bounds = schedule.open(encoder, self.passes.bounds);
                self.bounds.record_stream(&mut bounds, streams);
                drop(bounds);
            }
            Phase::Entries => {
                let mut entries = schedule.open(encoder, self.passes.entries);
                self.entries.record_stream(&mut entries, streams);
                drop(entries);
            }
            Phase::Deform => {
                let mut substeps = schedule.open(encoder, self.passes.substeps);
                for _ in 0..frame.params.soft_substeps {
                    self.reset.record_stream(&mut substeps, streams);
                    self.integrate.record_stream(&mut substeps, streams);
                    for _ in 0..frame.params.soft_iterations {
                        self.elements.record_stream(&mut substeps, streams);
                        self.gather.record_stream(&mut substeps, streams);
                    }
                }
                drop(substeps);

                let mut collide = schedule.open(encoder, self.passes.collide);
                self.collide.record_stream(&mut collide, streams);
                drop(collide);

                let mut apply = schedule.open(encoder, self.passes.apply);
                self.apply
                    .record_rows(&mut apply, streams, Count::Bodies.rows(&frame.params));
                drop(apply);
            }
            _ => {}
        }
    }
}
