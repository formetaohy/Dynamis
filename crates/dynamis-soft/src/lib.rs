mod capacity;
mod domain;
mod streams;

pub use capacity::{SoftCapacity, SoftInputs, capacity, floor, plan};
pub use domain::{SoftDomain, SoftWork};
pub use streams::{SoftDemand, SoftStream, SoftStreams};

use dynamis_abi::{Count, StepParamsRecord};
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::Resources;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{Execution, Stage, domain_passes};
use dynamis_shader::{CORE, rows};
use dynamis_state::StateStream;

const PARTICLE_SHAPE: &[&str] = &[include_str!("../shaders/particle_shape.wgsl")];
const PARTICLE_REACH: &[&str] = &[include_str!("../shaders/particle_reach.wgsl")];
const PARTICLE_KERNEL: &[&str] = &[include_str!("../shaders/particle_kernel.wgsl")];
const ELEMENT_SAMPLE: &[&str] = &[include_str!("../shaders/element_sample.wgsl")];
const SOFT_REACTION: &[&str] = &[include_str!("../shaders/soft_reaction.wgsl")];
const SOFT_ANCHOR: &[&str] = &[include_str!("../shaders/soft_anchor.wgsl")];
const SOFT_FILTER: &[&str] = &[include_str!("../shaders/soft_filter.wgsl")];
const SOFT_ATTACH: &[&str] = &[
    include_str!("../shaders/soft_reaction.wgsl"),
    include_str!("../shaders/soft_anchor.wgsl"),
];

fn particle_index() -> Vec<&'static str> {
    let mut fragments = dynamis_shader::GRID_INDEX.to_vec();
    fragments.extend_from_slice(PARTICLE_SHAPE);
    fragments
}

fn particle_fluid_index() -> Vec<&'static str> {
    let mut fragments = dynamis_shader::GRID_INDEX.to_vec();
    fragments.extend_from_slice(PARTICLE_REACH);
    fragments.extend_from_slice(PARTICLE_KERNEL);
    fragments.extend_from_slice(SOFT_FILTER);
    fragments
}

fn particle_collide_index() -> Vec<&'static str> {
    let mut fragments = dynamis_shader::GEOMETRY_INDEX.to_vec();
    fragments.extend_from_slice(PARTICLE_REACH);
    fragments.extend_from_slice(SOFT_FILTER);
    fragments
}

fn particle_wake_index() -> Vec<&'static str> {
    let mut fragments = dynamis_shader::GRID_INDEX.to_vec();
    fragments.extend_from_slice(PARTICLE_REACH);
    fragments.extend_from_slice(SOFT_FILTER);
    fragments
}

domain_passes!(
    SoftPasses,
    soft_bounds => Execution::AWAKE => &[],
    soft_entries => Execution::AWAKE => &["soft_bounds", "prepare"],
    soft_settle => Execution::AWAKE => &["ccd_apply"],
    soft_substeps => Execution::AWAKE => &["soft_settle"],
    soft_apply => Execution::AWAKE => &["soft_substeps"],
);

#[derive(Clone, Copy, Debug)]
pub struct SoftFrame {
    pub params: StepParamsRecord,
    pub material: bool,
}

pub struct Soft {
    passes: SoftPasses,
    bounds: Stage,
    entries: Stage,
    activity: Stage,
    wake: Stage,
    attach_wake: Stage,
    rest: Stage,
    integrate: Stage,
    reset: Stage,
    elements: Stage,
    gather: Stage,
    attachment: Stage,
    density: Stage,
    pressure: Stage,
    detect: Stage,
    resolve: Stage,
    material: Stage,
    apply: Stage,
}

impl Soft {
    pub fn new(context: &GpuContext, streams: &impl Resources, passes: SoftPasses) -> Self {
        let particles = SoftStream::Particles;
        let bodies = SoftStream::BodyStates;
        let index = particle_index();
        let reach = particle_fluid_index();
        let collide = particle_collide_index();
        let wake_fragments = particle_wake_index();
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
            activity: Stage::build(
                context,
                "soft_activity",
                rows(
                    context,
                    include_str!("../shaders/soft_activity.wgsl"),
                    CORE,
                    Count::Particles.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("bodies", bodies.whole()),
                ],
                &[],
            ),
            wake: Stage::build(
                context,
                "soft_wake",
                rows(
                    context,
                    include_str!("../shaders/soft_wake.wgsl"),
                    &wake_fragments,
                    Count::Particles.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("bodies", bodies.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("colliders", StateStream::Colliders.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
            attach_wake: Stage::build(
                context,
                "soft_attach_wake",
                rows(
                    context,
                    include_str!("../shaders/soft_attach_wake.wgsl"),
                    SOFT_ANCHOR,
                    Count::Attachments.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("attachments", SoftStream::Attachments.whole()),
                    ("particles", particles.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("row_of_body", StateStream::BodyRowOfId.whole()),
                    ("bodies", bodies.whole()),
                ],
                &[],
            ),
            rest: Stage::build(
                context,
                "soft_rest",
                rows(
                    context,
                    include_str!("../shaders/soft_rest.wgsl"),
                    CORE,
                    Count::SoftBodies.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("bodies", bodies.whole()),
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
                    ("bodies", bodies.whole()),
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
                    ELEMENT_SAMPLE,
                    Count::Elements.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("elements", SoftStream::Elements.whole()),
                    ("contributions", SoftStream::Contributions.whole()),
                    ("bodies", bodies.whole()),
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
                    ("elements", SoftStream::Elements.whole()),
                    ("bodies", bodies.whole()),
                ],
                &[],
            ),
            attachment: Stage::build(
                context,
                "soft_attach",
                rows(
                    context,
                    include_str!("../shaders/soft_attach.wgsl"),
                    SOFT_ATTACH,
                    Count::Attachments.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("attachments", SoftStream::Attachments.whole()),
                    ("particles", particles.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("row_of_body", StateStream::BodyRowOfId.whole()),
                    ("reactions", SoftStream::Reactions.whole()),
                    ("bodies", bodies.whole()),
                ],
                &[],
            ),
            density: Stage::build(
                context,
                "soft_density",
                rows(
                    context,
                    include_str!("../shaders/soft_density.wgsl"),
                    &reach,
                    Count::Particles.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("pressure", SoftStream::Pressure.whole()),
                    ("bodies", bodies.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
            pressure: Stage::build(
                context,
                "soft_pressure",
                rows(
                    context,
                    include_str!("../shaders/soft_pressure.wgsl"),
                    &reach,
                    Count::Particles.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("pressure", SoftStream::Pressure.whole()),
                    ("bodies", bodies.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
            detect: Stage::build(
                context,
                "soft_collide_detect",
                rows(
                    context,
                    include_str!("../shaders/soft_collide_detect.wgsl"),
                    &collide,
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
                    ("bodies", bodies.whole()),
                ],
                &dynamis_state::shape_resources(),
            ),
            resolve: Stage::build(
                context,
                "soft_collide_resolve",
                rows(
                    context,
                    include_str!("../shaders/soft_collide_resolve.wgsl"),
                    SOFT_REACTION,
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
                    ("bodies", bodies.whole()),
                ],
                &[],
            ),
            material: Stage::build(
                context,
                "soft_material",
                rows(
                    context,
                    include_str!("../shaders/soft_material.wgsl"),
                    ELEMENT_SAMPLE,
                    Count::Elements.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("elements", SoftStream::Elements.whole()),
                    ("bodies", bodies.whole()),
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
        &mut self,
        pass: u32,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &SoftFrame,
    ) -> bool {
        let particles = Count::Particles.rows(&frame.params);
        let elements = Count::Elements.rows(&frame.params);
        let bodies = Count::SoftBodies.rows(&frame.params);
        let attachments = Count::Attachments.rows(&frame.params);
        if pass == self.passes.soft_bounds {
            self.bounds.record_rows(recorder, streams, particles);
        } else if pass == self.passes.soft_entries {
            self.entries.record_rows(recorder, streams, particles);
        } else if pass == self.passes.soft_settle {
            self.activity.record_rows(recorder, streams, particles);
            self.wake.record_rows(recorder, streams, particles);
            self.attach_wake.record_rows(recorder, streams, attachments);
            self.rest.record_rows(recorder, streams, bodies);
        } else if pass == self.passes.soft_substeps {
            for _ in 0..frame.params.soft_substeps {
                self.reset.record_rows(recorder, streams, elements);
                self.integrate.record_rows(recorder, streams, particles);
                for _ in 0..frame.params.soft_iterations {
                    self.elements.record_rows(recorder, streams, elements);
                    self.gather.record_rows(recorder, streams, particles);
                    self.density.record_rows(recorder, streams, particles);
                    self.pressure.record_rows(recorder, streams, particles);
                }
                self.attachment.record_rows(recorder, streams, attachments);
                self.detect.record_rows(recorder, streams, particles);
                self.resolve.record_rows(recorder, streams, particles);
                if frame.material {
                    self.material.record_rows(recorder, streams, elements);
                }
            }
        } else if pass == self.passes.soft_apply {
            self.apply
                .record_rows(recorder, streams, Count::Bodies.rows(&frame.params));
        } else {
            return false;
        }
        true
    }
}
