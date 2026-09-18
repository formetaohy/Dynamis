use super::emit_impacts::EmitImpacts;
use super::streams::RigidStream;
use super::{CONTACT_ROW, IDENTITY};
use crate::RigidFrame;
use crate::sort;
use dynamis_abi::Count;
use dynamis_abi::{
    COUNTER_ARCHIVED, COUNTER_BREAKS, COUNTER_CONTACTS, COUNTER_EVENTS, COUNTER_REFUSED_EVENTS,
    COUNTER_REFUSED_RESTING, COUNTER_RESTING, COUNTER_RESTING_GATHER, COUNTER_RESTING_INDEX,
    COUNTER_RESTING_PENDING, COUNTER_SLEPT, COUNTER_WOKE_DEFERRED,
};
use dynamis_gpu::ResourceSource;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::Execution;
use dynamis_pass::{PassRuntime, Stage};
use dynamis_shader::{CORE, Extent, JOINTS, rows, stream, workgroups};
use dynamis_sort::RadixSort;
use dynamis_state::StateStream;

pub const OBSERVED_JOINTS_GATE: u32 = 1;

pub const OBSERVED_JOINTS_EXECUTION: Execution =
    Execution::PUBLISH.and(Execution::gate(OBSERVED_JOINTS_GATE));

pub struct Commit {
    impacts: EmitImpacts,
    thaw_contacts: Stage,
    freeze_contacts: Stage,
    contact_archive: Stage,
    constraint_breaks: Stage,
    archive_count_sync: Stage,
    static_wake_clear: Stage,
}

pub struct RestingGather {
    resting_gather: Stage,
}

pub struct RestingIndex {
    resting_commit: Stage,
    sort: RadixSort,
}

pub struct Observe {
    observe: Stage,
}

pub struct ObserveJoints {
    observe_joints: Stage,
}

impl PassRuntime<RigidFrame> for Commit {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            impacts: EmitImpacts::build(context, streams),
            thaw_contacts: Stage::build(
                context,
                "thaw_contacts",
                stream(
                    context,
                    include_str!("../shaders/thaw_contacts.wgsl"),
                    CONTACT_ROW,
                    "work",
                    Extent::slot(COUNTER_RESTING, "resting_count", "resting_live"),
                ),
                streams,
                &[
                    ("resting", RigidStream::RestingContacts.whole()),
                    ("resting_live", RigidStream::RestingLive.whole()),
                    ("resting_next", RigidStream::RestingNext.whole()),
                    ("resting_free", RigidStream::RestingFree.whole()),
                    ("resting_count", dynamis_state::counter(COUNTER_RESTING)),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("row_of_body", StateStream::BodyRowOfId.whole()),
                    ("body_activity", RigidStream::BodyActivity.whole()),
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
                    ("events", RigidStream::Events.whole()),
                    ("event_count", dynamis_state::counter(COUNTER_EVENTS)),
                    ("spillover", dynamis_state::counter(COUNTER_REFUSED_EVENTS)),
                    ("params", StateStream::Params.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
            freeze_contacts: Stage::build(
                context,
                "freeze_contacts",
                stream(
                    context,
                    include_str!("../shaders/freeze_contacts.wgsl"),
                    CORE,
                    "work",
                    Extent::slot(COUNTER_CONTACTS, "contact_count", "contacts"),
                ),
                streams,
                &[
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("resting", RigidStream::RestingContacts.whole()),
                    ("resting_live", RigidStream::RestingLive.whole()),
                    ("resting_count", dynamis_state::counter(COUNTER_RESTING)),
                    ("spillover", dynamis_state::counter(COUNTER_REFUSED_RESTING)),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("resting_next", RigidStream::RestingNext.whole()),
                    ("resting_free", RigidStream::RestingFree.whole()),
                    ("collider_owners", StateStream::ColliderOwners.whole()),
                ],
                &[],
            ),
            contact_archive: Stage::build(
                context,
                "contact_archive",
                stream(
                    context,
                    include_str!("../shaders/contact_archive.wgsl"),
                    CORE,
                    "work",
                    Extent::slot(COUNTER_CONTACTS, "contact_count", "contacts"),
                ),
                streams,
                &[
                    ("contacts", RigidStream::Contacts.whole()),
                    ("archive", RigidStream::ContactArchive.whole()),
                    ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
                ],
                &[],
            ),
            constraint_breaks: Stage::build(
                context,
                "constraint_breaks",
                rows(
                    context,
                    include_str!("../shaders/constraint_breaks.wgsl"),
                    dynamis_shader::COUNTERS,
                    Count::Constraints.bound(),
                ),
                streams,
                &[
                    ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                    ("constraint_breaks", StateStream::ConstraintBreaks.whole()),
                    ("break_count", dynamis_state::counter(COUNTER_BREAKS)),
                    ("counters", StateStream::Counters.whole()),
                    ("params", StateStream::Params.whole()),
                ],
                &[],
            ),
            archive_count_sync: Stage::build(
                context,
                "archive_count_sync",
                workgroups(
                    context,
                    include_str!("../shaders/archive_count_sync.wgsl"),
                    CORE,
                ),
                streams,
                &[
                    ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
                    ("archive_count", dynamis_state::counter(COUNTER_ARCHIVED)),
                    ("contacts", RigidStream::Contacts.whole()),
                ],
                &[],
            ),
            static_wake_clear: Stage::build(
                context,
                "static_wake_clear",
                rows(
                    context,
                    include_str!("../shaders/static_wake_clear.wgsl"),
                    dynamis_shader::COUNTERS,
                    Count::Bodies.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
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
        frame: &RigidFrame,
    ) {
        self.impacts.record(recorder, streams, frame);
        self.thaw_contacts.record_stream(recorder, streams);
        self.contact_archive.record_stream(recorder, streams);
        self.archive_count_sync
            .record_workgroups(recorder, streams, 1);
        self.static_wake_clear.record_rows(
            recorder,
            streams,
            Count::Bodies.rows(&frame.params, &frame.rows),
        );
        self.freeze_contacts.record_stream(recorder, streams);
        self.constraint_breaks.record_rows(
            recorder,
            streams,
            Count::Constraints.rows(&frame.params, &frame.rows),
        );
    }
}

impl PassRuntime<RigidFrame> for RestingGather {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            resting_gather: Stage::build(
                context,
                "resting_gather",
                stream(
                    context,
                    include_str!("../shaders/resting_gather.wgsl"),
                    IDENTITY,
                    "work",
                    Extent::slot(COUNTER_RESTING, "resting_count", "resting_live"),
                ),
                streams,
                &[
                    ("resting", RigidStream::RestingContacts.whole()),
                    ("resting_count", dynamis_state::counter(COUNTER_RESTING)),
                    ("resting_live", RigidStream::RestingLive.whole()),
                    ("index_major", RigidStream::RestingIndexMajor.whole()),
                    ("index_minor", RigidStream::RestingIndexMinor.whole()),
                    ("index_slots", RigidStream::RestingIndexSlots.whole()),
                    ("gathered", dynamis_state::counter(COUNTER_RESTING_GATHER)),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        _: &RigidFrame,
    ) {
        self.resting_gather.record_stream(recorder, streams);
    }
}

impl PassRuntime<RigidFrame> for RestingIndex {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            resting_commit: Stage::build(
                context,
                "resting_commit",
                workgroups(
                    context,
                    include_str!("../shaders/resting_commit.wgsl"),
                    CORE,
                ),
                streams,
                &[
                    ("slept", dynamis_state::counter(COUNTER_SLEPT)),
                    ("gathered", dynamis_state::counter(COUNTER_RESTING_GATHER)),
                    ("index_count", dynamis_state::counter(COUNTER_RESTING_INDEX)),
                    (
                        "deferred_woke",
                        dynamis_state::counter(COUNTER_WOKE_DEFERRED),
                    ),
                    ("pending", dynamis_state::counter(COUNTER_RESTING_PENDING)),
                ],
                &[],
            ),
            sort: RadixSort::new(context),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        self.resting_commit.record_workgroups(recorder, streams, 1);
        let words = frame.shape.body_id_words;
        let channels = sort::keyed(
            streams,
            dynamis_state::counter(COUNTER_RESTING_GATHER),
            RigidStream::RestingIndexMajor.whole(),
            RigidStream::RestingIndexMinor.whole(),
            RigidStream::RestingIndexSlots.whole(),
        );
        let plan = dynamis_sort::units_for(streams.measured(COUNTER_RESTING_GATHER).unwrap_or(0));
        self.sort.sort(recorder, &channels, plan, words, words);
    }
}

impl PassRuntime<RigidFrame> for Observe {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            observe: Stage::build(
                context,
                "observe",
                rows(
                    context,
                    include_str!("../shaders/observe.wgsl"),
                    CORE,
                    Count::Observed.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("observed_ids", StateStream::ObservedIds.whole()),
                    ("row_of_body", StateStream::BodyRowOfId.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("observed_states", StateStream::ObservedStates.whole()),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        self.observe
            .record_rows(recorder, streams, frame.observed_count);
    }
}

impl PassRuntime<RigidFrame> for ObserveJoints {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            observe_joints: Stage::build(
                context,
                "observe_joints",
                rows(
                    context,
                    include_str!("../shaders/observe_joints.wgsl"),
                    JOINTS,
                    Count::ObservedJoints.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("observed_joint_ids", StateStream::ObservedJointIds.whole()),
                    ("row_of_constraint", StateStream::ConstraintRowOfId.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    (
                        "constraint_descs",
                        StateStream::ConstraintDescriptors.whole(),
                    ),
                    ("constraint_rows", RigidStream::ConstraintRows.whole()),
                    ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                    (
                        "observed_joint_states",
                        StateStream::ObservedJointStates.whole(),
                    ),
                    (
                        "observed_joint_runtimes",
                        StateStream::ObservedJointRuntimes.whole(),
                    ),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        self.observe_joints
            .record_rows(recorder, streams, frame.observed_joints);
    }
}
