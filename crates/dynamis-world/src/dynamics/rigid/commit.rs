use super::Count;
use super::streams::RigidStream;
use crate::dynamics::Frame;
use crate::dynamics::engine::Stage;
use crate::dynamics::scene::SceneStream;
use crate::dynamics::shader;
use crate::dynamics::shader::{CONTACT, CORE, GEOMETRY_INDEX, IDENTITY};
use crate::dynamics::streams::Streams;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{
    COUNTER_ARCHIVED, COUNTER_CONTACTS, COUNTER_EVENTS, COUNTER_RESTING, COUNTER_RESTING_GATHER,
    COUNTER_RESTING_INDEX, COUNTER_RESTING_PENDING, COUNTER_SLEPT, COUNTER_SPILLOVER_EVENTS,
    COUNTER_SPILLOVER_RESTING, COUNTER_WOKE_DEFERRED,
};
use dynamis_sort::RadixSort;

pub(super) struct Commit {
    thaw_contacts: Stage,
    freeze_contacts: Stage,
    resting_gather: Stage,
    resting_commit: Stage,
    contact_archive: Stage,
    archive_count_sync: Stage,
    static_wake_clear: Stage,
    query: Stage,
}

impl Commit {
    pub(super) fn build(context: &GpuContext, streams: &Streams) -> Self {
        Self {
            thaw_contacts: Stage::build(
                context,
                "thaw_contacts",
                shader::stream(
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
                    ("resting_count", streams.scene.counter(COUNTER_RESTING)),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("row_of_body", SceneStream::BodyRowOfId.whole()),
                    ("body_activity", RigidStream::BodyActivity.whole()),
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_count", streams.scene.counter(COUNTER_CONTACTS)),
                    ("events", RigidStream::Events.whole()),
                    ("event_count", streams.scene.counter(COUNTER_EVENTS)),
                    ("spillover", streams.scene.counter(COUNTER_SPILLOVER_EVENTS)),
                    ("params", SceneStream::Params.whole()),
                ],
                &[],
            ),
            resting_gather: Stage::build(
                context,
                "resting_gather",
                shader::stream(
                    context,
                    include_str!("../shaders/resting_gather.wgsl"),
                    IDENTITY,
                    "work",
                    RigidStream::RestingContacts,
                ),
                streams,
                &[
                    ("resting", RigidStream::RestingContacts.whole()),
                    ("resting_count", streams.scene.counter(COUNTER_RESTING)),
                    ("resting_live", RigidStream::RestingLive.whole()),
                    ("index_major", RigidStream::RestingIndexMajor.whole()),
                    ("index_minor", RigidStream::RestingIndexMinor.whole()),
                    ("index_slots", RigidStream::RestingIndexSlots.whole()),
                    ("gathered", streams.scene.counter(COUNTER_RESTING_GATHER)),
                ],
                &[],
            ),
            resting_commit: Stage::build(
                context,
                "resting_commit",
                shader::workgroups(
                    context,
                    include_str!("../shaders/resting_commit.wgsl"),
                    CORE,
                ),
                streams,
                &[
                    ("slept", streams.scene.counter(COUNTER_SLEPT)),
                    ("gathered", streams.scene.counter(COUNTER_RESTING_GATHER)),
                    ("index_count", streams.scene.counter(COUNTER_RESTING_INDEX)),
                    (
                        "deferred_woke",
                        streams.scene.counter(COUNTER_WOKE_DEFERRED),
                    ),
                    ("pending", streams.scene.counter(COUNTER_RESTING_PENDING)),
                ],
                &[],
            ),
            freeze_contacts: Stage::build(
                context,
                "freeze_contacts",
                shader::stream(
                    context,
                    include_str!("../shaders/freeze_contacts.wgsl"),
                    CORE,
                    "work",
                    RigidStream::Contacts,
                ),
                streams,
                &[
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_count", streams.scene.counter(COUNTER_CONTACTS)),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("resting", RigidStream::RestingContacts.whole()),
                    ("resting_live", RigidStream::RestingLive.whole()),
                    ("resting_count", streams.scene.counter(COUNTER_RESTING)),
                    (
                        "spillover",
                        streams.scene.counter(COUNTER_SPILLOVER_RESTING),
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
                shader::stream(
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
                    ("contact_count", streams.scene.counter(COUNTER_CONTACTS)),
                ],
                &[],
            ),
            archive_count_sync: Stage::build(
                context,
                "archive_count_sync",
                shader::workgroups(
                    context,
                    include_str!("../shaders/archive_count_sync.wgsl"),
                    CORE,
                ),
                streams,
                &[
                    ("contact_count", streams.scene.counter(COUNTER_CONTACTS)),
                    ("archive_count", streams.scene.counter(COUNTER_ARCHIVED)),
                    ("contacts", RigidStream::Contacts.whole()),
                ],
                &[],
            ),
            static_wake_clear: Stage::build(
                context,
                "static_wake_clear",
                shader::rows(
                    context,
                    include_str!("../shaders/static_wake_clear.wgsl"),
                    CORE,
                    Count::Bodies,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("wake_flags", RigidStream::WakeFlags.whole()),
                ],
                &[],
            ),
            query: Stage::build(
                context,
                "query",
                shader::workgroups(
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
                    ("entry_keys", RigidStream::GridEntryKeys.whole()),
                    ("entry_colliders", RigidStream::GridEntryColliders.whole()),
                    ("counters", SceneStream::Counters.whole()),
                    ("query_results", SceneStream::QueryResults.whole()),
                    ("params", SceneStream::Params.whole()),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
                ],
                &streams.scene.shape_resources(),
            ),
        }
    }

    pub(super) fn record_query(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &Streams,
        frame: &Frame,
    ) {
        self.query
            .record_workgroups(recorder, streams, frame.query_count);
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, streams: &Streams, frame: &Frame) {
        self.thaw_contacts.record_stream(recorder, streams);
        self.contact_archive.record_stream(recorder, streams);
        self.archive_count_sync
            .record_workgroups(recorder, streams, 1);
        self.static_wake_clear
            .record_rows(recorder, streams, Count::Bodies.rows(&frame.params));
        self.freeze_contacts.record_stream(recorder, streams);
        self.record_query(recorder, streams, frame);
    }

    pub(super) fn record_gather(&self, recorder: &mut ComputeRecorder, streams: &Streams) {
        self.resting_gather.record_stream(recorder, streams);
    }

    pub(super) fn record_index(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &Streams,
        sort: &RadixSort,
    ) {
        self.resting_commit.record_workgroups(recorder, streams, 1);
        let words = streams.scene.collider_words();
        let channels = streams.sort_keyed(
            streams.scene.counter(COUNTER_RESTING_GATHER),
            RigidStream::RestingIndexMajor.whole(),
            RigidStream::RestingIndexMinor.whole(),
            RigidStream::RestingIndexSlots.whole(),
        );
        sort.sort(
            recorder,
            &channels,
            words,
            words,
            streams.rigid.resting_capacity(),
        );
    }
}
