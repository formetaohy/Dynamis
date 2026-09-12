use super::FrameParams;
use super::stage::{CONTACT, CORE, GEOMETRY, IDENTITY, Stage, shape_resources, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{
    COUNTER_ARCHIVED, COUNTER_CONTACTS, COUNTER_ENTRIES, COUNTER_EVENTS, COUNTER_LARGE,
    COUNTER_RESTING, COUNTER_RESTING_GATHER, COUNTER_RESTING_INDEX, COUNTER_RESTING_PENDING,
    COUNTER_SLEPT, COUNTER_SPILLOVER_EVENTS, COUNTER_SPILLOVER_RESTING, COUNTER_WOKE_DEFERRED,
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
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            thaw_contacts: Stage::build(
                context,
                "thaw_contacts",
                include_str!("shaders/thaw_contacts.wgsl"),
                per_row,
                CONTACT,
                &[
                    ("resting", whole(&buffers.contacts.resting)),
                    ("resting_live", whole(&buffers.contacts.resting_live)),
                    ("resting_next", whole(&buffers.contacts.resting_next)),
                    ("resting_free", whole(&buffers.contacts.resting_free)),
                    ("resting_count", buffers.counter(COUNTER_RESTING)),
                    ("body_states", whole(&buffers.bodies.states)),
                    ("row_of_body", whole(&buffers.bodies.row_of_body)),
                    ("body_activity", whole(&buffers.bodies.activity)),
                    ("contacts", whole(&buffers.contacts.manifolds)),
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
                include_str!("shaders/resting_gather.wgsl"),
                per_row,
                IDENTITY,
                &[
                    ("resting", whole(&buffers.contacts.resting)),
                    ("resting_count", buffers.counter(COUNTER_RESTING)),
                    ("resting_live", whole(&buffers.contacts.resting_live)),
                    ("index_major", whole(&buffers.contacts.resting_index.major)),
                    ("index_minor", whole(&buffers.contacts.resting_index.minor)),
                    (
                        "index_slots",
                        whole(&buffers.contacts.resting_index.payload),
                    ),
                    ("gathered", buffers.counter(COUNTER_RESTING_GATHER)),
                ],
                &[],
            ),
            resting_commit: Stage::build(
                context,
                "resting_commit",
                include_str!("shaders/resting_commit.wgsl"),
                per_row,
                CORE,
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
                include_str!("shaders/freeze_contacts.wgsl"),
                per_row,
                CORE,
                &[
                    ("contacts", whole(&buffers.contacts.manifolds)),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("body_states", whole(&buffers.bodies.states)),
                    ("resting", whole(&buffers.contacts.resting)),
                    ("resting_live", whole(&buffers.contacts.resting_live)),
                    ("resting_count", buffers.counter(COUNTER_RESTING)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_RESTING)),
                    ("body_descs", whole(&buffers.bodies.descriptors)),
                    ("resting_next", whole(&buffers.contacts.resting_next)),
                    ("resting_free", whole(&buffers.contacts.resting_free)),
                    ("collider_owners", whole(&buffers.bodies.collider_owners)),
                ],
                &[],
            ),
            contact_archive: Stage::build(
                context,
                "contact_archive",
                include_str!("shaders/contact_archive.wgsl"),
                per_row,
                CORE,
                &[
                    ("contacts", whole(&buffers.contacts.manifolds)),
                    ("archive", whole(&buffers.contacts.archive)),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                ],
                &[],
            ),
            archive_count_sync: Stage::build(
                context,
                "archive_count_sync",
                include_str!("shaders/archive_count_sync.wgsl"),
                per_row,
                CORE,
                &[
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("archive_count", buffers.counter(COUNTER_ARCHIVED)),
                    ("contacts", whole(&buffers.contacts.manifolds)),
                ],
                &[],
            ),
            static_wake_clear: Stage::build(
                context,
                "static_wake_clear",
                include_str!("shaders/static_wake_clear.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("wake_flags", whole(&buffers.islands.wake_flags)),
                ],
                &[],
            ),
            query: Stage::build(
                context,
                "query",
                include_str!("shaders/queries.wgsl"),
                per_row,
                GEOMETRY,
                &[
                    ("queries", whole(&buffers.queries.records)),
                    ("body_states", whole(&buffers.bodies.states)),
                    ("body_descs", whole(&buffers.bodies.descriptors)),
                    ("colliders", whole(&buffers.bodies.colliders)),
                    ("aabbs", whole(&buffers.bodies.aabbs)),
                    ("entry_cells", whole(&buffers.contacts.entries.cells)),
                    (
                        "entry_colliders",
                        whole(&buffers.contacts.entries.colliders),
                    ),
                    ("entry_count", buffers.counter(COUNTER_ENTRIES)),
                    ("query_results", whole(&buffers.queries.results)),
                    ("large_bodies", whole(&buffers.contacts.large_bodies)),
                    ("large_count", buffers.counter(COUNTER_LARGE)),
                    ("params", whole(&buffers.params)),
                    ("collider_owners", whole(&buffers.bodies.collider_owners)),
                ],
                &shape_resources(buffers),
            ),
        }
    }

    pub(super) fn record_query(&self, recorder: &mut ComputeRecorder, query_count: u32) {
        self.query.record_workgroups(recorder, query_count);
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        params: &FrameParams,
    ) {
        let contacts = buffers.contact_capacity();
        self.thaw_contacts
            .record_stride(recorder, buffers.resting_capacity());
        self.contact_archive.record_stride(recorder, contacts);
        self.archive_count_sync.record_workgroups(recorder, 1);
        self.static_wake_clear.record(recorder, params.body_count);
        self.freeze_contacts.record_stride(recorder, contacts);
        if params.query_count > 0 {
            self.record_query(recorder, params.query_count);
        }
    }

    pub(super) fn record_gather(&self, recorder: &mut ComputeRecorder, buffers: &WorldBuffers) {
        self.resting_gather
            .record_stride(recorder, buffers.resting_capacity());
    }

    pub(super) fn record_index(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        sort: &RadixSort,
    ) {
        self.resting_commit.record_workgroups(recorder, 1);
        let words = buffers.collider_words();
        let channels = buffers.sort_keyed(
            buffers.counter(COUNTER_RESTING_GATHER),
            &buffers.contacts.resting_index.major,
            &buffers.contacts.resting_index.minor,
            &buffers.contacts.resting_index.payload,
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
