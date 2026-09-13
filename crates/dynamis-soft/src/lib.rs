mod capacity;
mod streams;

pub use capacity::{SoftCapacity, capacity, floor, plan};
pub use streams::{DOMAIN, SoftDemand, SoftStream, SoftStreams};

use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::GpuContext;
use dynamis_kernels::{CORE, GEOMETRY_INDEX, rows};
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
    detect: Phase::Deform => "soft_collide_detect",
    resolve: Phase::Deform => "soft_collide_resolve",
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
    detect: Stage,
    resolve: Stage,
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
                rows(
                    context,
                    include_str!("../shaders/particle_bounds.wgsl"),
                    PARTICLE_SHAPE,
                    Count::Particles.field(),
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
                rows(
                    context,
                    include_str!("../shaders/particle_entries.wgsl"),
                    &index,
                    Count::Particles.field(),
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
                rows(
                    context,
                    include_str!("../shaders/soft_integrate.wgsl"),
                    CORE,
                    Count::Particles.field(),
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
                rows(
                    context,
                    include_str!("../shaders/soft_reset.wgsl"),
                    CORE,
                    Count::Elements.field(),
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
                rows(
                    context,
                    include_str!("../shaders/soft_elements.wgsl"),
                    CORE,
                    Count::Elements.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("elements", SoftStream::Elements.whole()),
                    ("contributions", SoftStream::Contributions.whole()),
                ],
                &[],
            ),
            gather: Stage::build(
                context,
                "soft_gather",
                rows(
                    context,
                    include_str!("../shaders/soft_gather.wgsl"),
                    CORE,
                    Count::Particles.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("contributions", SoftStream::Contributions.whole()),
                    ("adjacency", SoftStream::Adjacency.whole()),
                ],
                &[],
            ),
            detect: Stage::build(
                context,
                "soft_collide_detect",
                rows(
                    context,
                    include_str!("../shaders/soft_collide_detect.wgsl"),
                    GEOMETRY_INDEX,
                    Count::Particles.field(),
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
                    ("contacts", SoftStream::Contacts.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &dynamis_state::shape_resources(),
            ),
            resolve: Stage::build(
                context,
                "soft_collide_resolve",
                rows(
                    context,
                    include_str!("../shaders/soft_collide_resolve.wgsl"),
                    CORE,
                    Count::Particles.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("contacts", SoftStream::Contacts.whole()),
                    ("reactions", SoftStream::Reactions.whole()),
                ],
                &[],
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
        let particles = Count::Particles.rows(&frame.params);
        let elements = Count::Elements.rows(&frame.params);
        match phase {
            Phase::Prepare => {
                let mut bounds = schedule.open(encoder, self.passes.bounds);
                self.bounds.record_rows(&mut bounds, streams, particles);
                drop(bounds);
            }
            Phase::Entries => {
                let mut entries = schedule.open(encoder, self.passes.entries);
                self.entries.record_rows(&mut entries, streams, particles);
                drop(entries);
            }
            Phase::Deform => {
                let mut substeps = schedule.open(encoder, self.passes.substeps);
                for _ in 0..frame.params.soft_substeps {
                    self.reset.record_rows(&mut substeps, streams, elements);
                    self.integrate
                        .record_rows(&mut substeps, streams, particles);
                    for _ in 0..frame.params.soft_iterations {
                        self.elements.record_rows(&mut substeps, streams, elements);
                        self.gather.record_rows(&mut substeps, streams, particles);
                    }
                }
                drop(substeps);

                let mut detect = schedule.open(encoder, self.passes.detect);
                self.detect.record_rows(&mut detect, streams, particles);
                drop(detect);

                let mut resolve = schedule.open(encoder, self.passes.resolve);
                self.resolve.record_rows(&mut resolve, streams, particles);
                drop(resolve);

                let mut apply = schedule.open(encoder, self.passes.apply);
                self.apply
                    .record_rows(&mut apply, streams, Count::Bodies.rows(&frame.params));
                drop(apply);
            }
            _ => {}
        }
    }
}
