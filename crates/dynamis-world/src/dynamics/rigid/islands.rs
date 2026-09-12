use super::Count;
use super::Frame;
use super::buffers::{RigidBuffers, StreamId};
use super::shader;
use super::shader::{CONTACT, CORE, IDENTITY};
use crate::dynamics::engine::Stage;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{
    COUNTER_ARCHIVED, COUNTER_CONTACTS, COUNTER_EVENTS, COUNTER_RESTING_INDEX,
    COUNTER_SPILLOVER_EVENTS,
};

pub(super) struct Islands {
    contact_relay: Stage,
    contact_begin: Stage,
    island_init: Stage,
    island_link_contacts: Stage,
    island_link_constraints: Stage,
    island_jump: Stage,
}

impl Islands {
    pub(super) fn build(context: &GpuContext, buffers: &RigidBuffers) -> Self {
        Self {
            contact_relay: Stage::build(
                context,
                "contact_relay",
                shader::stream(
                    context,
                    include_str!("shaders/contact_relay.wgsl"),
                    CONTACT,
                    "work",
                    StreamId::ContactArchive,
                ),
                buffers,
                &[
                    ("archive", StreamId::ContactArchive.whole()),
                    ("archive_count", buffers.counter(COUNTER_ARCHIVED)),
                    ("contacts", StreamId::Contacts.whole()),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("contact_matched", StreamId::ContactMatched.whole()),
                    ("events", StreamId::Events.whole()),
                    ("event_count", buffers.counter(COUNTER_EVENTS)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                    ("params", StreamId::Params.whole()),
                    ("row_of_body", StreamId::BodyRowOfId.whole()),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("body_descs", StreamId::BodyDescriptors.whole()),
                ],
                &[],
            ),
            contact_begin: Stage::build(
                context,
                "contact_begin",
                shader::stream(
                    context,
                    include_str!("shaders/contact_begin.wgsl"),
                    CONTACT,
                    "work",
                    StreamId::Contacts,
                ),
                buffers,
                &[
                    ("contacts", StreamId::Contacts.whole()),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("contact_matched", StreamId::ContactMatched.whole()),
                    ("events", StreamId::Events.whole()),
                    ("event_count", buffers.counter(COUNTER_EVENTS)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                    ("params", StreamId::Params.whole()),
                    ("resting", StreamId::RestingContacts.whole()),
                    ("resting_live", StreamId::RestingLive.whole()),
                    ("resting_major", StreamId::RestingIndexMajor.whole()),
                    ("resting_minor", StreamId::RestingIndexMinor.whole()),
                    ("resting_slots", StreamId::RestingIndexSlots.whole()),
                    ("resting_index", buffers.counter(COUNTER_RESTING_INDEX)),
                ],
                &[],
            ),
            island_init: Stage::build(
                context,
                "island_init",
                shader::rows(
                    context,
                    include_str!("shaders/island_init.wgsl"),
                    CORE,
                    Count::Dynamic,
                ),
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("island_parents", StreamId::IslandParents.whole()),
                    ("island_state", StreamId::IslandState.whole()),
                ],
                &[],
            ),
            island_link_contacts: Stage::build(
                context,
                "island_link_contacts",
                shader::stream(
                    context,
                    include_str!("shaders/island_link_contacts.wgsl"),
                    IDENTITY,
                    "work",
                    StreamId::Contacts,
                ),
                buffers,
                &[
                    ("body_states", StreamId::BodyStates.whole()),
                    ("body_descs", StreamId::BodyDescriptors.whole()),
                    ("contacts", StreamId::Contacts.whole()),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("island_parents", StreamId::IslandParents.whole()),
                    ("wake_flags", StreamId::WakeFlags.whole()),
                    ("params", StreamId::Params.whole()),
                    ("collider_owners", StreamId::ColliderOwners.whole()),
                ],
                &[],
            ),
            island_link_constraints: Stage::build(
                context,
                "island_link_constraints",
                shader::rows(
                    context,
                    include_str!("shaders/island_link_constraints.wgsl"),
                    CORE,
                    Count::Constraints,
                ),
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("body_descs", StreamId::BodyDescriptors.whole()),
                    ("constraint_runtime", StreamId::ConstraintRuntime.whole()),
                    ("island_parents", StreamId::IslandParents.whole()),
                    ("wake_flags", StreamId::WakeFlags.whole()),
                    ("constraint_rows", StreamId::ConstraintRows.whole()),
                ],
                &[],
            ),
            island_jump: Stage::build(
                context,
                "island_jump",
                shader::rows(
                    context,
                    include_str!("shaders/island_jump.wgsl"),
                    CORE,
                    Count::Dynamic,
                ),
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("island_parents", StreamId::IslandParents.whole()),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        self.contact_relay.record_stream(recorder, buffers);
        self.contact_begin.record_stream(recorder, buffers);
        self.island_init
            .record_rows(recorder, buffers, Count::Dynamic.rows(&frame.params));
        self.island_link_contacts.record_stream(recorder, buffers);
        self.island_link_constraints.record_rows(
            recorder,
            buffers,
            Count::Constraints.rows(&frame.params),
        );
        for _ in 0..frame.island_rounds {
            self.island_jump
                .record_rows(recorder, buffers, Count::Dynamic.rows(&frame.params));
        }
    }
}
