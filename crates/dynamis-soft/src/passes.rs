use crate::streams::SoftStream;
use dynamis_abi::{
    COUNTER_REFUSED_SOFT_EVENTS, COUNTER_SOFT_EVENTS, Count, RowStreams, StepParamsRecord,
};
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::{ComputeRecorder, GpuContext, ResourceSource};
use dynamis_pass::{Execution, PassRuntime, Stage, domain_passes};
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

fn particle_counter_shape() -> Vec<&'static str> {
    let mut fragments = dynamis_shader::COUNTERS.to_vec();
    fragments.extend_from_slice(PARTICLE_SHAPE);
    fragments
}

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

fn particle_fact_index() -> Vec<&'static str> {
    vec![dynamis_shader::COUNTER_ACCESS, dynamis_shader::CONTACT_FACT]
}

fn particle_wake_index() -> Vec<&'static str> {
    let mut fragments = dynamis_shader::GRID_INDEX.to_vec();
    fragments.extend_from_slice(PARTICLE_REACH);
    fragments.extend_from_slice(SOFT_FILTER);
    fragments
}

#[derive(Clone, Copy, Debug)]
pub struct SoftFrame {
    pub params: StepParamsRecord,
    pub rows: RowStreams,
    pub material: bool,
}

impl SoftFrame {
    fn particles(&self) -> u32 {
        Count::Particles.rows(&self.params, &self.rows)
    }

    fn elements(&self) -> u32 {
        Count::Elements.rows(&self.params, &self.rows)
    }

    fn bodies(&self) -> u32 {
        Count::SoftBodies.rows(&self.params, &self.rows)
    }

    fn attachments(&self) -> u32 {
        Count::Attachments.rows(&self.params, &self.rows)
    }
}

pub struct UpdateSoftBounds {
    bounds: Stage,
}

impl PassRuntime<SoftFrame> for UpdateSoftBounds {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            bounds: Stage::build(
                context,
                "soft_bounds",
                rows(
                    context,
                    include_str!("../shaders/particle_bounds.wgsl"),
                    &particle_counter_shape(),
                    Count::Particles.bound(),
                ),
                streams,
                &[
                    ("counters", StateStream::Counters.whole()),
                    ("params", StateStream::Params.whole()),
                    ("particles", SoftStream::Particles.whole()),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &SoftFrame,
    ) {
        self.bounds
            .record_rows(recorder, streams, frame.particles());
    }
}

pub struct EmitSoftEntries {
    entries: Stage,
}

impl PassRuntime<SoftFrame> for EmitSoftEntries {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            entries: Stage::build(
                context,
                "soft_entries",
                rows(
                    context,
                    include_str!("../shaders/particle_entries.wgsl"),
                    &particle_index(),
                    Count::Particles.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("particles", SoftStream::Particles.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &SoftFrame,
    ) {
        self.entries
            .record_rows(recorder, streams, frame.particles());
    }
}

pub struct ApplySoftInputs {
    clear: Stage,
    body_edits: Stage,
    particle_edits: Stage,
}

impl PassRuntime<SoftFrame> for ApplySoftInputs {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        let bodies = SoftStream::BodyStates;
        Self {
            clear: Stage::build(
                context,
                "soft_input_clear",
                rows(
                    context,
                    include_str!("../shaders/soft_input_clear.wgsl"),
                    CORE,
                    Count::SoftBodies.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("bodies", bodies.whole()),
                ],
                &[],
            ),
            body_edits: Stage::build(
                context,
                "soft_body_edits",
                rows(
                    context,
                    include_str!("../shaders/soft_body_edits.wgsl"),
                    CORE,
                    Count::SoftBodyEdits.bound(),
                ),
                streams,
                &[
                    ("row_streams", StateStream::RowStreams.whole()),
                    ("bodies", bodies.whole()),
                    ("edits", SoftStream::BodyEdits.whole()),
                ],
                &[],
            ),
            particle_edits: Stage::build(
                context,
                "soft_particle_edits",
                rows(
                    context,
                    include_str!("../shaders/soft_edits.wgsl"),
                    CORE,
                    Count::SoftEdits.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("row_streams", StateStream::RowStreams.whole()),
                    ("particles", SoftStream::Particles.whole()),
                    ("edits", SoftStream::Edits.whole()),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &SoftFrame,
    ) {
        self.clear.record_rows(recorder, streams, frame.bodies());
        self.body_edits.record_rows(
            recorder,
            streams,
            Count::SoftBodyEdits.rows(&frame.params, &frame.rows),
        );
        self.particle_edits.record_rows(
            recorder,
            streams,
            Count::SoftEdits.rows(&frame.params, &frame.rows),
        );
    }
}

pub const WAKE_ALL_GATE: u32 = 0;

pub const WAKE_ALL_EXECUTION: Execution = Execution::STEP
    .and(Execution::AWAKE)
    .and(Execution::gate(WAKE_ALL_GATE));

pub struct WakeAll {
    wake: Stage,
}

impl PassRuntime<SoftFrame> for WakeAll {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            wake: Stage::build(
                context,
                "soft_wake_all",
                rows(
                    context,
                    include_str!("../shaders/soft_wake_all.wgsl"),
                    CORE,
                    Count::SoftBodies.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("bodies", SoftStream::BodyStates.whole()),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &SoftFrame,
    ) {
        self.wake.record_rows(recorder, streams, frame.bodies());
    }
}

pub struct SoftSettle {
    activity: Stage,
    wake: Stage,
    attach_wake: Stage,
    rest: Stage,
}

impl PassRuntime<SoftFrame> for SoftSettle {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        let particles = SoftStream::Particles;
        let bodies = SoftStream::BodyStates;
        Self {
            activity: Stage::build(
                context,
                "soft_activity",
                rows(
                    context,
                    include_str!("../shaders/soft_activity.wgsl"),
                    CORE,
                    Count::Particles.bound(),
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
                    &particle_wake_index(),
                    Count::Particles.bound(),
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
                    Count::Attachments.bound(),
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
                    ("wake_flags", StateStream::WakeFlags.whole()),
                ],
                &[],
            ),
            rest: Stage::build(
                context,
                "soft_rest",
                rows(
                    context,
                    include_str!("../shaders/soft_rest.wgsl"),
                    dynamis_shader::COUNTERS,
                    Count::SoftBodies.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("bodies", bodies.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &SoftFrame,
    ) {
        let particles = frame.particles();
        self.activity.record_rows(recorder, streams, particles);
        self.wake.record_rows(recorder, streams, particles);
        self.attach_wake
            .record_rows(recorder, streams, frame.attachments());
        self.rest.record_rows(recorder, streams, frame.bodies());
    }
}

pub struct SolveSoftSubsteps {
    reset: Stage,
    integrate: Stage,
    elements: Stage,
    gather: Stage,
    attachment: Stage,
    density: Stage,
    pressure: Stage,
    detect: Stage,
    resolve: Stage,
    material: Stage,
}

impl PassRuntime<SoftFrame> for SolveSoftSubsteps {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        let particles = SoftStream::Particles;
        let bodies = SoftStream::BodyStates;
        let reach = particle_fluid_index();
        Self {
            reset: Stage::build(
                context,
                "soft_reset",
                rows(
                    context,
                    include_str!("../shaders/soft_reset.wgsl"),
                    CORE,
                    Count::Elements.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("elements", SoftStream::Elements.whole()),
                ],
                &[],
            ),
            integrate: Stage::build(
                context,
                "soft_integrate",
                rows(
                    context,
                    include_str!("../shaders/soft_integrate.wgsl"),
                    &[dynamis_shader::FIELD],
                    Count::Particles.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("bodies", bodies.whole()),
                    ("fields", StateStream::Fields.whole()),
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
                    Count::Elements.bound(),
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
                    Count::Particles.bound(),
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
                    Count::Attachments.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("attachments", SoftStream::Attachments.whole()),
                    ("particles", particles.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("row_of_body", StateStream::BodyRowOfId.whole()),
                    ("reactions", StateStream::BodyReactions.whole()),
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
                    Count::Particles.bound(),
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
                    Count::Particles.bound(),
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
                    &particle_collide_index(),
                    Count::Particles.bound(),
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
                    Count::Particles.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("particles", particles.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("contacts", SoftStream::Contacts.whole()),
                    ("reactions", StateStream::BodyReactions.whole()),
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
                    Count::Elements.bound(),
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
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &SoftFrame,
    ) {
        let particles = frame.particles();
        let elements = frame.elements();
        let attachments = frame.attachments();
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
    }
}

pub struct EmitContactFacts {
    announce: Stage,
}

impl PassRuntime<SoftFrame> for EmitContactFacts {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        let particles = SoftStream::Particles.whole();
        let bodies = SoftStream::BodyStates.whole();
        Self {
            announce: Stage::build(
                context,
                "soft_contact_facts",
                rows(
                    context,
                    include_str!("../shaders/soft_contact_facts.wgsl"),
                    &particle_fact_index(),
                    Count::Particles.bound(),
                ),
                streams,
                &[
                    ("contacts", SoftStream::Contacts.whole()),
                    ("bodies", bodies),
                    ("colliders", StateStream::Colliders.whole()),
                    ("particles", particles),
                    ("events", SoftStream::Events.whole()),
                    ("event_count", dynamis_state::counter(COUNTER_SOFT_EVENTS)),
                    (
                        "spillover",
                        dynamis_state::counter(COUNTER_REFUSED_SOFT_EVENTS),
                    ),
                    ("counters", StateStream::Counters.whole()),
                    ("params", StateStream::Params.whole()),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &SoftFrame,
    ) {
        self.announce
            .record_rows(recorder, streams, frame.particles());
    }
}

domain_passes!(
    SoftPasses,
    SoftRuntime,
    SoftFrame,
    apply_soft_inputs: ApplySoftInputs => Execution::STEP.and(Execution::AWAKE) => &["apply_commands"],
    update_soft_bounds: UpdateSoftBounds => Execution::INDEXING => &["apply_soft_inputs"],
    emit_soft_entries: EmitSoftEntries => Execution::INDEXING => &["update_soft_bounds", "prepare"],
    soft_wake_all: WakeAll => WAKE_ALL_EXECUTION => &["apply_soft_inputs"],
    soft_settle: SoftSettle => Execution::AWAKE => &["apply_soft_inputs", "soft_wake_all"],
    solve_soft_substeps: SolveSoftSubsteps => Execution::AWAKE => &["soft_settle"],
    emit_contact_facts: EmitContactFacts => Execution::AWAKE => &["solve_soft_substeps"],
);
