use super::streams::RigidStream;
use super::{CONTACT, IDENTITY};
use crate::RigidFrame;
use crate::sort;
use dynamis_abi::Count;
use dynamis_abi::{
    COUNTER_ARCHIVED, COUNTER_BREAKS, COUNTER_CONTACTS, COUNTER_EVENTS, COUNTER_RESTING,
    COUNTER_RESTING_GATHER, COUNTER_RESTING_INDEX, COUNTER_RESTING_PENDING, COUNTER_SLEPT,
    COUNTER_SPILLOVER_EVENTS, COUNTER_SPILLOVER_RESTING, COUNTER_WOKE_DEFERRED,
};
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernel::{CORE, GEOMETRY_INDEX, JOINTS, rows, stream, workgroups};
use dynamis_pass::Resources;
use dynamis_pass::Stage;
use dynamis_sort::RadixSort;
use dynamis_state::StateStream;

pub struct Commit {
    thaw_contacts: Stage,
    freeze_contacts: Stage,
    resting_gather: Stage,
    resting_commit: Stage,
    contact_archive: Stage,
    constraint_breaks: Stage,
    joint_states: Stage,
    archive_count_sync: Stage,
    static_wake_clear: Stage,
    query: Stage,
    observe: Stage,
}

impl Commit {
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            thaw_contacts: Stage::build(
                context,
                "thaw_contacts",
                stream(
                    context,
                    include_str!("../shaders/thaw_contacts.wgsl"),
                    CONTACT,
                    "work",
                    RigidStream::RestingContacts,
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
                    (
                        "spillover",
                        dynamis_state::counter(COUNTER_SPILLOVER_EVENTS),
                    ),
                    ("params", StateStream::Params.whole()),
                ],
                &[],
            ),
            resting_gather: Stage::build(
                context,
                "resting_gather",
                stream(
                    context,
                    include_str!("../shaders/resting_gather.wgsl"),
                    IDENTITY,
                    "work",
                    RigidStream::RestingContacts,
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
            freeze_contacts: Stage::build(
                context,
                "freeze_contacts",
                stream(
                    context,
                    include_str!("../shaders/freeze_contacts.wgsl"),
                    CORE,
                    "work",
                    RigidStream::Contacts,
                ),
                streams,
                &[
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("resting", RigidStream::RestingContacts.whole()),
                    ("resting_live", RigidStream::RestingLive.whole()),
                    ("resting_count", dynamis_state::counter(COUNTER_RESTING)),
                    (
                        "spillover",
                        dynamis_state::counter(COUNTER_SPILLOVER_RESTING),
                    ),
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
                    RigidStream::Contacts,
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
                    CORE,
                    Count::Constraints.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                    ("constraint_breaks", StateStream::ConstraintBreaks.whole()),
                    ("break_count", dynamis_state::counter(COUNTER_BREAKS)),
                ],
                &[],
            ),
            joint_states: Stage::build(
                context,
                "joint_states",
                rows(
                    context,
                    include_str!("../shaders/joint_states.wgsl"),
                    JOINTS,
                    Count::Constraints.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    (
                        "constraint_descs",
                        StateStream::ConstraintDescriptors.whole(),
                    ),
                    ("constraint_rows", RigidStream::ConstraintRows.whole()),
                    ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                    ("joint_states", RigidStream::JointStates.whole()),
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
                    CORE,
                    Count::Bodies.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                ],
                &[],
            ),
            query: Stage::build(
                context,
                "query",
                workgroups(
                    context,
                    include_str!("../shaders/queries.wgsl"),
                    GEOMETRY_INDEX,
                ),
                streams,
                &[
                    ("queries", StateStream::QueryRecords.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("colliders", StateStream::Colliders.whole()),
                    ("aabbs", RigidStream::ColliderAabbs.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                    ("query_results", StateStream::QueryResults.whole()),
                    ("params", StateStream::Params.whole()),
                    ("collider_owners", StateStream::ColliderOwners.whole()),
                ],
                &dynamis_state::shape_resources(),
            ),
            observe: Stage::build(
                context,
                "observe",
                rows(
                    context,
                    include_str!("../shaders/observe.wgsl"),
                    CORE,
                    Count::Observed.field(),
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

    pub fn record_query(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.query
            .record_workgroups(recorder, streams, frame.query_count);
    }

    pub fn record_observe(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        count: u32,
    ) {
        self.observe.record_rows(recorder, streams, count);
    }

    pub fn record(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.thaw_contacts.record_stream(recorder, streams);
        self.contact_archive.record_stream(recorder, streams);
        self.archive_count_sync
            .record_workgroups(recorder, streams, 1);
        self.static_wake_clear
            .record_rows(recorder, streams, Count::Bodies.rows(&frame.params));
        self.freeze_contacts.record_stream(recorder, streams);
        self.constraint_breaks.record_rows(
            recorder,
            streams,
            Count::Constraints.rows(&frame.params),
        );
        self.joint_states
            .record_rows(recorder, streams, Count::Constraints.rows(&frame.params));
        self.record_query(recorder, streams, frame);
        self.record_observe(recorder, streams, frame.observed_count);
    }

    pub fn record_gather(&self, recorder: &mut ComputeRecorder, streams: &impl Resources) {
        self.resting_gather.record_stream(recorder, streams);
    }

    pub fn record_index(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
        sort: &RadixSort,
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
        sort.sort(recorder, &channels, words, words);
    }
}
