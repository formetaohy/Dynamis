use super::Count;
use super::Frame;
use super::buffers::RigidBuffers;
use super::shader;
use super::shader::{CONTACT, CORE, IDENTITY};
use crate::dynamics::engine::{Stage, whole};
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
                    buffers.archive_capacity(),
                ),
                &[
                    ("archive", whole(&buffers.contact_archive)),
                    ("archive_count", buffers.counter(COUNTER_ARCHIVED)),
                    ("contacts", whole(&buffers.contacts)),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("contact_matched", whole(&buffers.contact_matched)),
                    ("events", whole(&buffers.events)),
                    ("event_count", buffers.counter(COUNTER_EVENTS)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                    ("params", whole(&buffers.params)),
                    ("row_of_body", whole(&buffers.body_row_of_id)),
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
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
                    buffers.contact_capacity(),
                ),
                &[
                    ("contacts", whole(&buffers.contacts)),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("contact_matched", whole(&buffers.contact_matched)),
                    ("events", whole(&buffers.events)),
                    ("event_count", buffers.counter(COUNTER_EVENTS)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                    ("params", whole(&buffers.params)),
                    ("resting", whole(&buffers.resting_contacts)),
                    ("resting_live", whole(&buffers.resting_live)),
                    ("resting_major", whole(&buffers.resting_index_major)),
                    ("resting_minor", whole(&buffers.resting_index_minor)),
                    ("resting_slots", whole(&buffers.resting_index_slots)),
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
                &[
                    ("params", whole(&buffers.params)),
                    ("island_parents", whole(&buffers.island_parents)),
                    ("island_state", whole(&buffers.island_state)),
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
                    buffers.contact_capacity(),
                ),
                &[
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("contacts", whole(&buffers.contacts)),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("island_parents", whole(&buffers.island_parents)),
                    ("wake_flags", whole(&buffers.wake_flags)),
                    ("params", whole(&buffers.params)),
                    ("collider_owners", whole(&buffers.collider_owners)),
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
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("constraint_runtime", whole(&buffers.constraint_runtime)),
                    ("island_parents", whole(&buffers.island_parents)),
                    ("wake_flags", whole(&buffers.wake_flags)),
                    ("constraint_rows", whole(&buffers.constraint_rows)),
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
                &[
                    ("params", whole(&buffers.params)),
                    ("island_parents", whole(&buffers.island_parents)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, frame: &Frame) {
        self.contact_relay.record_stream(recorder);
        self.contact_begin.record_stream(recorder);
        self.island_init
            .record_rows(recorder, Count::Dynamic.rows(&frame.params));
        self.island_link_contacts.record_stream(recorder);
        self.island_link_constraints
            .record_rows(recorder, Count::Constraints.rows(&frame.params));
        for _ in 0..frame.island_rounds {
            self.island_jump
                .record_rows(recorder, Count::Dynamic.rows(&frame.params));
        }
    }
}
