use super::streams::RigidStream;
use super::{CONTACT, IDENTITY};
use crate::sort;
use crate::streams::resting_capacity;
use dynamis_abi::{
    COUNTER_ARCHIVED, COUNTER_BREAKS, COUNTER_CONTACTS, COUNTER_EVENTS, COUNTER_RESTING,
    COUNTER_RESTING_GATHER, COUNTER_RESTING_INDEX, COUNTER_RESTING_PENDING, COUNTER_SLEPT,
    COUNTER_SPILLOVER_EVENTS, COUNTER_SPILLOVER_RESTING, COUNTER_WOKE_DEFERRED,
};
use dynamis_broadphase::BroadphaseStream;
use dynamis_engine::Resources;
use dynamis_engine::Stage;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernels::{CORE, GEOMETRY_INDEX, rows, stream, workgroups};
use dynamis_scene::Count;
use dynamis_scene::Frame;
use dynamis_scene::SceneStream;
use dynamis_sort::RadixSort;

pub struct Commit {
    thaw_contacts: Stage,
    freeze_contacts: Stage,
    resting_gather: Stage,
    resting_commit: Stage,
    contact_archive: Stage,
    constraint_breaks: Stage,
    archive_count_sync: Stage,
    static_wake_clear: Stage,
    query: Stage,
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
                    ("resting_count", dynamis_scene::counter(COUNTER_RESTING)),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("row_of_body", SceneStream::BodyRowOfId.whole()),
                    ("body_activity", RigidStream::BodyActivity.whole()),
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_count", dynamis_scene::counter(COUNTER_CONTACTS)),
                    ("events", RigidStream::Events.whole()),
                    ("event_count", dynamis_scene::counter(COUNTER_EVENTS)),
                    (
                        "spillover",
                        dynamis_scene::counter(COUNTER_SPILLOVER_EVENTS),
                    ),
                    ("params", SceneStream::Params.whole()),
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
                    ("resting_count", dynamis_scene::counter(COUNTER_RESTING)),
                    ("resting_live", RigidStream::RestingLive.whole()),
                    ("index_major", RigidStream::RestingIndexMajor.whole()),
                    ("index_minor", RigidStream::RestingIndexMinor.whole()),
                    ("index_slots", RigidStream::RestingIndexSlots.whole()),
                    ("gathered", dynamis_scene::counter(COUNTER_RESTING_GATHER)),
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
                    ("slept", dynamis_scene::counter(COUNTER_SLEPT)),
                    ("gathered", dynamis_scene::counter(COUNTER_RESTING_GATHER)),
                    ("index_count", dynamis_scene::counter(COUNTER_RESTING_INDEX)),
                    (
                        "deferred_woke",
                        dynamis_scene::counter(COUNTER_WOKE_DEFERRED),
                    ),
                    ("pending", dynamis_scene::counter(COUNTER_RESTING_PENDING)),
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
                    ("contact_count", dynamis_scene::counter(COUNTER_CONTACTS)),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("resting", RigidStream::RestingContacts.whole()),
                    ("resting_live", RigidStream::RestingLive.whole()),
                    ("resting_count", dynamis_scene::counter(COUNTER_RESTING)),
                    (
                        "spillover",
                        dynamis_scene::counter(COUNTER_SPILLOVER_RESTING),
                    ),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("resting_next", RigidStream::RestingNext.whole()),
                    ("resting_free", RigidStream::RestingFree.whole()),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
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
                    ("contact_count", dynamis_scene::counter(COUNTER_CONTACTS)),
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
                    ("params", SceneStream::Params.whole()),
                    ("constraint_runtime", SceneStream::ConstraintRuntime.whole()),
                    ("constraint_breaks", SceneStream::ConstraintBreaks.whole()),
                    ("break_count", dynamis_scene::counter(COUNTER_BREAKS)),
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
                    ("contact_count", dynamis_scene::counter(COUNTER_CONTACTS)),
                    ("archive_count", dynamis_scene::counter(COUNTER_ARCHIVED)),
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
                    ("params", SceneStream::Params.whole()),
                    ("wake_flags", SceneStream::WakeFlags.whole()),
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
                    ("queries", SceneStream::QueryRecords.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("colliders", SceneStream::Colliders.whole()),
                    ("aabbs", RigidStream::ColliderAabbs.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", SceneStream::Counters.whole()),
                    ("query_results", SceneStream::QueryResults.whole()),
                    ("params", SceneStream::Params.whole()),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
                ],
                &dynamis_scene::shape_resources(),
            ),
        }
    }

    pub fn record_query(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &Frame,
    ) {
        self.query
            .record_workgroups(recorder, streams, frame.query_count);
    }

    pub fn record(&self, recorder: &mut ComputeRecorder, streams: &impl Resources, frame: &Frame) {
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
        self.record_query(recorder, streams, frame);
    }

    pub fn record_gather(&self, recorder: &mut ComputeRecorder, streams: &impl Resources) {
        self.resting_gather.record_stream(recorder, streams);
    }

    pub fn record_index(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        sort: &RadixSort,
    ) {
        self.resting_commit.record_workgroups(recorder, streams, 1);
        let words = dynamis_scene::collider_words(streams);
        let channels = sort::keyed(
            streams,
            dynamis_scene::counter(COUNTER_RESTING_GATHER),
            RigidStream::RestingIndexMajor.whole(),
            RigidStream::RestingIndexMinor.whole(),
            RigidStream::RestingIndexSlots.whole(),
        );
        sort.sort(recorder, &channels, words, words, resting_capacity(streams));
    }
}
