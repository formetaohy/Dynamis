use super::Count;
use super::Frame;
use super::buffers::RigidBuffers;
use super::shader;
use super::shader::{CONTACT, CORE, GEOMETRY_INDEX, IDENTITY};
use crate::dynamics::engine::{Stage, whole};
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
                    buffers.resting_capacity(),
                ),
                &[
                    ("resting", whole(&buffers.resting_contacts)),
                    ("resting_live", whole(&buffers.resting_live)),
                    ("resting_next", whole(&buffers.resting_next)),
                    ("resting_free", whole(&buffers.resting_free)),
                    ("resting_count", buffers.counter(COUNTER_RESTING)),
                    ("body_states", whole(&buffers.body_states)),
                    ("row_of_body", whole(&buffers.body_row_of_id)),
                    ("body_activity", whole(&buffers.body_activity)),
                    ("contacts", whole(&buffers.contacts)),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("events", whole(&buffers.events)),
                    ("event_count", buffers.counter(COUNTER_EVENTS)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                    ("params", whole(&buffers.params)),
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
                    buffers.resting_capacity(),
                ),
                &[
                    ("resting", whole(&buffers.resting_contacts)),
                    ("resting_count", buffers.counter(COUNTER_RESTING)),
                    ("resting_live", whole(&buffers.resting_live)),
                    ("index_major", whole(&buffers.resting_index_major)),
                    ("index_minor", whole(&buffers.resting_index_minor)),
                    ("index_slots", whole(&buffers.resting_index_slots)),
                    ("gathered", buffers.counter(COUNTER_RESTING_GATHER)),
                ],
                &[],
            ),
            resting_commit: Stage::build(
                context,
                "resting_commit",
                shader::workgroups(context, include_str!("shaders/resting_commit.wgsl"), CORE),
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
                    buffers.contact_capacity(),
                ),
                &[
                    ("contacts", whole(&buffers.contacts)),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("body_states", whole(&buffers.body_states)),
                    ("resting", whole(&buffers.resting_contacts)),
                    ("resting_live", whole(&buffers.resting_live)),
                    ("resting_count", buffers.counter(COUNTER_RESTING)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_RESTING)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("resting_next", whole(&buffers.resting_next)),
                    ("resting_free", whole(&buffers.resting_free)),
                    ("collider_owners", whole(&buffers.collider_owners)),
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
                    buffers.contact_capacity(),
                ),
                &[
                    ("contacts", whole(&buffers.contacts)),
                    ("archive", whole(&buffers.contact_archive)),
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
                &[
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("archive_count", buffers.counter(COUNTER_ARCHIVED)),
                    ("contacts", whole(&buffers.contacts)),
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
                &[
                    ("params", whole(&buffers.params)),
                    ("wake_flags", whole(&buffers.wake_flags)),
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
                &[
                    ("queries", whole(&buffers.query_records)),
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("colliders", whole(&buffers.colliders)),
                    ("aabbs", whole(&buffers.collider_aabbs)),
                    ("entry_keys", whole(&buffers.grid_entry_keys)),
                    ("entry_colliders", whole(&buffers.grid_entry_colliders)),
                    ("counters", whole(&buffers.counters)),
                    ("query_results", whole(&buffers.query_results)),
                    ("params", whole(&buffers.params)),
                    ("collider_owners", whole(&buffers.collider_owners)),
                ],
                &buffers.shape_resources(),
            ),
        }
    }

    pub(super) fn record_query(&self, recorder: &mut ComputeRecorder, frame: &Frame) {
        self.query.record_workgroups(recorder, frame.query_count);
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, frame: &Frame) {
        self.thaw_contacts.record_stream(recorder);
        self.contact_archive.record_stream(recorder);
        self.archive_count_sync.record_workgroups(recorder, 1);
        self.static_wake_clear
            .record_rows(recorder, Count::Bodies.rows(&frame.params));
        self.freeze_contacts.record_stream(recorder);
        self.record_query(recorder, frame);
    }

    pub(super) fn record_gather(&self, recorder: &mut ComputeRecorder) {
        self.resting_gather.record_stream(recorder);
    }

    pub(super) fn record_index(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        sort: &RadixSort,
    ) {
        self.resting_commit.record_workgroups(recorder, 1);
        let words = buffers.collider_words();
        let channels = buffers.sort_keyed(
            buffers.counter(COUNTER_RESTING_GATHER),
            &buffers.resting_index_major,
            &buffers.resting_index_minor,
            &buffers.resting_index_slots,
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
