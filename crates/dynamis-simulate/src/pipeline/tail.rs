use super::FrameParams;
use super::dispatch::{CONTACT_ARCHIVE, EVENTS_END, FREEZE_CONTACTS, THAW_CONTACTS};
use super::stage::{RO, RW, Stage, UNIFORM, shape_resources, whole};
use crate::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{
    COUNTER_CONTACTS, COUNTER_ENTRIES, COUNTER_EVENTS, COUNTER_LARGE, COUNTER_PREV_CONTACTS,
    COUNTER_RESTING, COUNTER_SPILLOVER_EVENTS, COUNTER_SPILLOVER_RESTING,
};

pub(super) struct Tail {
    events_end: Stage,
    thaw_contacts: Stage,
    freeze_contacts: Stage,
    contact_archive: Stage,
    prev_count_sync: Stage,
    constraints_warm_end: Stage,
    static_wake_clear: Stage,
    query: Stage,
}

impl Tail {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            events_end: Stage::build(
                context,
                "events_end",
                include_str!("../shaders/events_end.wgsl"),
                per_row,
                &[
                    (RO, whole(&buffers.contacts.previous)),
                    (RW, buffers.counter(COUNTER_PREV_CONTACTS)),
                    (RO, whole(&buffers.contacts.manifolds)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RW, whole(&buffers.events)),
                    (RW, buffers.counter(COUNTER_EVENTS)),
                    (RW, buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                ],
                &[],
            ),
            thaw_contacts: Stage::build(
                context,
                "thaw_contacts",
                include_str!("../shaders/thaw_contacts.wgsl"),
                per_row,
                &[
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.contacts.resting)),
                    (RW, whole(&buffers.contacts.resting_live)),
                    (RW, buffers.counter(COUNTER_RESTING)),
                    (RW, whole(&buffers.contacts.resting_next)),
                    (RW, whole(&buffers.contacts.resting_free)),
                    (RO, whole(&buffers.bodies.descriptors)),
                ],
                &[],
            ),
            freeze_contacts: Stage::build(
                context,
                "freeze_contacts",
                include_str!("../shaders/freeze_contacts.wgsl"),
                per_row,
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
                include_str!("../shaders/contact_archive.wgsl"),
                per_row,
                &[
                    (RO, whole(&buffers.contacts.manifolds)),
                    (RW, whole(&buffers.contacts.previous)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                ],
                &[],
            ),
            prev_count_sync: Stage::build(
                context,
                "prev_count_sync",
                include_str!("../shaders/prev_count_sync.wgsl"),
                per_row,
                &[
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RW, buffers.counter(COUNTER_PREV_CONTACTS)),
                    (RO, whole(&buffers.contacts.manifolds)),
                ],
                &[],
            ),
            constraints_warm_end: Stage::build(
                context,
                "constraints_warm_end",
                include_str!("../shaders/constraints_warm_end.wgsl"),
                per_row,
                &[
                    (RO, whole(&buffers.constraints.descriptors)),
                    (RW, whole(&buffers.constraints.runtime)),
                    (UNIFORM, whole(&buffers.params)),
                ],
                &[],
            ),
            static_wake_clear: Stage::build(
                context,
                "static_wake_clear",
                include_str!("../shaders/static_wake_clear.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.islands.wake_flags)),
                ],
                &[],
            ),
            query: Stage::build(
                context,
                "query",
                include_str!("../shaders/queries.wgsl"),
                per_row,
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
        self.events_end
            .record_indirect(recorder, &buffers.dispatch, EVENTS_END);
        self.contact_archive
            .record_indirect(recorder, &buffers.dispatch, CONTACT_ARCHIVE);
        self.prev_count_sync.record_workgroups(recorder, 1);
        self.constraints_warm_end
            .record(recorder, params.constraint_count);
        self.static_wake_clear.record(recorder, params.body_count);
        self.thaw_contacts
            .record_indirect(recorder, &buffers.dispatch, THAW_CONTACTS);
        self.freeze_contacts
            .record_indirect(recorder, &buffers.dispatch, FREEZE_CONTACTS);
        if params.query_count > 0 {
            self.record_query(recorder, params.query_count);
        }
    }
}
