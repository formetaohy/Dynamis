use super::Count;
use super::Frame;
use super::buffers::{RigidBuffers, StreamId};
use super::shader;
use super::shader::{CONTACT, CORE, GEOMETRY_INDEX, IDENTITY};
use crate::dynamics::engine::Stage;
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
    pub(super) fn build(context: &GpuContext, buffers: &RigidBuffers) -> Self {
        Self {
            thaw_contacts: Stage::build(
                context,
                "thaw_contacts",
                shader::stream(
                    context,
                    include_str!("shaders/thaw_contacts.wgsl"),
                    CONTACT,
                    "work",
                    StreamId::RestingContacts,
                ),
                buffers,
                &[
                    ("resting", StreamId::RestingContacts.whole()),
                    ("resting_live", StreamId::RestingLive.whole()),
                    ("resting_next", StreamId::RestingNext.whole()),
                    ("resting_free", StreamId::RestingFree.whole()),
                    ("resting_count", buffers.counter(COUNTER_RESTING)),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("row_of_body", StreamId::BodyRowOfId.whole()),
                    ("body_activity", StreamId::BodyActivity.whole()),
                    ("contacts", StreamId::Contacts.whole()),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("events", StreamId::Events.whole()),
                    ("event_count", buffers.counter(COUNTER_EVENTS)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                    ("params", StreamId::Params.whole()),
                ],
                &[],
            ),
            resting_gather: Stage::build(
                context,
                "resting_gather",
                shader::stream(
                    context,
                    include_str!("shaders/resting_gather.wgsl"),
                    IDENTITY,
                    "work",
                    StreamId::RestingContacts,
                ),
                buffers,
                &[
                    ("resting", StreamId::RestingContacts.whole()),
                    ("resting_count", buffers.counter(COUNTER_RESTING)),
                    ("resting_live", StreamId::RestingLive.whole()),
                    ("index_major", StreamId::RestingIndexMajor.whole()),
                    ("index_minor", StreamId::RestingIndexMinor.whole()),
                    ("index_slots", StreamId::RestingIndexSlots.whole()),
                    ("gathered", buffers.counter(COUNTER_RESTING_GATHER)),
                ],
                &[],
            ),
            resting_commit: Stage::build(
                context,
                "resting_commit",
                shader::workgroups(context, include_str!("shaders/resting_commit.wgsl"), CORE),
                buffers,
                &[
                    ("slept", buffers.counter(COUNTER_SLEPT)),
                    ("gathered", buffers.counter(COUNTER_RESTING_GATHER)),
                    ("index_count", buffers.counter(COUNTER_RESTING_INDEX)),
                    ("deferred_woke", buffers.counter(COUNTER_WOKE_DEFERRED)),
                    ("pending", buffers.counter(COUNTER_RESTING_PENDING)),
                ],
                &[],
            ),
            freeze_contacts: Stage::build(
                context,
                "freeze_contacts",
                shader::stream(
                    context,
                    include_str!("shaders/freeze_contacts.wgsl"),
                    CORE,
                    "work",
                    StreamId::Contacts,
                ),
                buffers,
                &[
                    ("contacts", StreamId::Contacts.whole()),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("resting", StreamId::RestingContacts.whole()),
                    ("resting_live", StreamId::RestingLive.whole()),
                    ("resting_count", buffers.counter(COUNTER_RESTING)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_RESTING)),
                    ("body_descs", StreamId::BodyDescriptors.whole()),
                    ("resting_next", StreamId::RestingNext.whole()),
                    ("resting_free", StreamId::RestingFree.whole()),
                    ("collider_owners", StreamId::ColliderOwners.whole()),
                ],
                &[],
            ),
            contact_archive: Stage::build(
                context,
                "contact_archive",
                shader::stream(
                    context,
                    include_str!("shaders/contact_archive.wgsl"),
                    CORE,
                    "work",
                    StreamId::Contacts,
                ),
                buffers,
                &[
                    ("contacts", StreamId::Contacts.whole()),
                    ("archive", StreamId::ContactArchive.whole()),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                ],
                &[],
            ),
            archive_count_sync: Stage::build(
                context,
                "archive_count_sync",
                shader::workgroups(
                    context,
                    include_str!("shaders/archive_count_sync.wgsl"),
                    CORE,
                ),
                buffers,
                &[
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("archive_count", buffers.counter(COUNTER_ARCHIVED)),
                    ("contacts", StreamId::Contacts.whole()),
                ],
                &[],
            ),
            static_wake_clear: Stage::build(
                context,
                "static_wake_clear",
                shader::rows(
                    context,
                    include_str!("shaders/static_wake_clear.wgsl"),
                    CORE,
                    Count::Bodies,
                ),
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("wake_flags", StreamId::WakeFlags.whole()),
                ],
                &[],
            ),
            query: Stage::build(
                context,
                "query",
                shader::workgroups(
                    context,
                    include_str!("shaders/queries.wgsl"),
                    GEOMETRY_INDEX,
                ),
                buffers,
                &[
                    ("queries", StreamId::QueryRecords.whole()),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("body_descs", StreamId::BodyDescriptors.whole()),
                    ("colliders", StreamId::Colliders.whole()),
                    ("aabbs", StreamId::ColliderAabbs.whole()),
                    ("entry_keys", StreamId::GridEntryKeys.whole()),
                    ("entry_colliders", StreamId::GridEntryColliders.whole()),
                    ("counters", StreamId::Counters.whole()),
                    ("query_results", StreamId::QueryResults.whole()),
                    ("params", StreamId::Params.whole()),
                    ("collider_owners", StreamId::ColliderOwners.whole()),
                ],
                &RigidBuffers::shape_resources(),
            ),
        }
    }

    pub(super) fn record_query(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        self.query
            .record_workgroups(recorder, buffers, frame.query_count);
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        self.thaw_contacts.record_stream(recorder, buffers);
        self.contact_archive.record_stream(recorder, buffers);
        self.archive_count_sync
            .record_workgroups(recorder, buffers, 1);
        self.static_wake_clear
            .record_rows(recorder, buffers, Count::Bodies.rows(&frame.params));
        self.freeze_contacts.record_stream(recorder, buffers);
        self.record_query(recorder, buffers, frame);
    }

    pub(super) fn record_gather(&self, recorder: &mut ComputeRecorder, buffers: &RigidBuffers) {
        self.resting_gather.record_stream(recorder, buffers);
    }

    pub(super) fn record_index(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        sort: &RadixSort,
    ) {
        self.resting_commit.record_workgroups(recorder, buffers, 1);
        let words = buffers.collider_words();
        let channels = buffers.sort_keyed(
            buffers.counter(COUNTER_RESTING_GATHER),
            StreamId::RestingIndexMajor.whole(),
            StreamId::RestingIndexMinor.whole(),
            StreamId::RestingIndexSlots.whole(),
        );
        sort.sort(
            recorder,
            &channels,
            words,
            words,
            buffers.resting_capacity(),
        );
    }
}
