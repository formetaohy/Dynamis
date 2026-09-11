use super::FrameParams;
use super::stage::{
    CONTACT, CORE, GEOMETRY, IDENTITY, RO, RW, Stage, UNIFORM, shape_resources, whole,
};
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
                    (RO, whole(&buffers.contacts.resting)),
                    (RW, whole(&buffers.contacts.resting_live)),
                    (RW, whole(&buffers.contacts.resting_next)),
                    (RW, whole(&buffers.contacts.resting_free)),
                    (RW, buffers.counter(COUNTER_RESTING)),
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.row_of_body)),
                    (RO, whole(&buffers.bodies.activity)),
                    (RO, whole(&buffers.contacts.manifolds)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RW, whole(&buffers.events)),
                    (RW, buffers.counter(COUNTER_EVENTS)),
                    (RW, buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                    (UNIFORM, whole(&buffers.params)),
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
                    (RO, whole(&buffers.contacts.resting)),
                    (RW, buffers.counter(COUNTER_RESTING)),
                    (RO, whole(&buffers.contacts.resting_live)),
                    (RW, whole(&buffers.contacts.resting_index.major)),
                    (RW, whole(&buffers.contacts.resting_index.minor)),
                    (RW, whole(&buffers.contacts.resting_index.payload)),
                    (RW, buffers.counter(COUNTER_RESTING_GATHER)),
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
                    (RW, buffers.counter(COUNTER_SLEPT)),
                    (RW, buffers.counter(COUNTER_RESTING_GATHER)),
                    (RW, buffers.counter(COUNTER_RESTING_INDEX)),
                    (RW, buffers.counter(COUNTER_WOKE_DEFERRED)),
                    (RW, buffers.counter(COUNTER_RESTING_PENDING)),
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
                    (RO, whole(&buffers.contacts.manifolds)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RO, whole(&buffers.bodies.states)),
                    (RW, whole(&buffers.contacts.resting)),
                    (RW, whole(&buffers.contacts.resting_live)),
                    (RW, buffers.counter(COUNTER_RESTING)),
                    (RW, buffers.counter(COUNTER_SPILLOVER_RESTING)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RW, whole(&buffers.contacts.resting_next)),
                    (RW, whole(&buffers.contacts.resting_free)),
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
                    (RO, whole(&buffers.contacts.manifolds)),
                    (RW, whole(&buffers.contacts.archive)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
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
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RW, buffers.counter(COUNTER_ARCHIVED)),
                    (RO, whole(&buffers.contacts.manifolds)),
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
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.islands.wake_flags)),
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
                    (RO, whole(&buffers.queries.records)),
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RO, whole(&buffers.bodies.colliders)),
                    (RO, whole(&buffers.bodies.aabbs)),
                    (RO, whole(&buffers.contacts.entries.cells)),
                    (RO, whole(&buffers.contacts.entries.colliders)),
                    (RW, buffers.counter(COUNTER_ENTRIES)),
                    (RW, whole(&buffers.queries.results)),
                    (RO, whole(&buffers.contacts.large_bodies)),
                    (RW, buffers.counter(COUNTER_LARGE)),
                    (UNIFORM, whole(&buffers.params)),
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
