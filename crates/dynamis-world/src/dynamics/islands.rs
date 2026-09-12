use super::FrameParams;
use super::stage::{CONTACT, CORE, IDENTITY, Stage, whole};
use crate::dynamics::buffers::WorldBuffers;
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
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            contact_relay: Stage::build(
                context,
                "contact_relay",
                include_str!("shaders/contact_relay.wgsl"),
                per_row,
                CONTACT,
                &[
                    ("archive", whole(&buffers.contacts.archive)),
                    ("archive_count", buffers.counter(COUNTER_ARCHIVED)),
                    ("contacts", whole(&buffers.contacts.manifolds)),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("contact_matched", whole(&buffers.contacts.contact_matched)),
                    ("events", whole(&buffers.events)),
                    ("event_count", buffers.counter(COUNTER_EVENTS)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                    ("params", whole(&buffers.params)),
                    ("row_of_body", whole(&buffers.bodies.row_of_body)),
                    ("body_states", whole(&buffers.bodies.states)),
                    ("body_descs", whole(&buffers.bodies.descriptors)),
                ],
                &[],
            ),
            contact_begin: Stage::build(
                context,
                "contact_begin",
                include_str!("shaders/contact_begin.wgsl"),
                per_row,
                CONTACT,
                &[
                    ("contacts", whole(&buffers.contacts.manifolds)),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("contact_matched", whole(&buffers.contacts.contact_matched)),
                    ("events", whole(&buffers.events)),
                    ("event_count", buffers.counter(COUNTER_EVENTS)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                    ("params", whole(&buffers.params)),
                    ("resting", whole(&buffers.contacts.resting)),
                    ("resting_live", whole(&buffers.contacts.resting_live)),
                    (
                        "resting_major",
                        whole(&buffers.contacts.resting_index.major),
                    ),
                    (
                        "resting_minor",
                        whole(&buffers.contacts.resting_index.minor),
                    ),
                    (
                        "resting_slots",
                        whole(&buffers.contacts.resting_index.payload),
                    ),
                    ("resting_index", buffers.counter(COUNTER_RESTING_INDEX)),
                ],
                &[],
            ),
            island_init: Stage::build(
                context,
                "island_init",
                include_str!("shaders/island_init.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("island_parents", whole(&buffers.islands.parents)),
                    ("island_state", whole(&buffers.islands.state)),
                ],
                &[],
            ),
            island_link_contacts: Stage::build(
                context,
                "island_link_contacts",
                include_str!("shaders/island_link_contacts.wgsl"),
                per_row,
                IDENTITY,
                &[
                    ("body_states", whole(&buffers.bodies.states)),
                    ("body_descs", whole(&buffers.bodies.descriptors)),
                    ("contacts", whole(&buffers.contacts.manifolds)),
                    ("contact_count", buffers.counter(COUNTER_CONTACTS)),
                    ("island_parents", whole(&buffers.islands.parents)),
                    ("wake_flags", whole(&buffers.islands.wake_flags)),
                    ("params", whole(&buffers.params)),
                ],
                &[],
            ),
            island_link_constraints: Stage::build(
                context,
                "island_link_constraints",
                include_str!("shaders/island_link_constraints.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.bodies.states)),
                    ("body_descs", whole(&buffers.bodies.descriptors)),
                    ("constraint_descs", whole(&buffers.constraints.descriptors)),
                    ("constraint_runtime", whole(&buffers.constraints.runtime)),
                    ("island_parents", whole(&buffers.islands.parents)),
                    ("wake_flags", whole(&buffers.islands.wake_flags)),
                ],
                &[],
            ),
            island_jump: Stage::build(
                context,
                "island_jump",
                include_str!("shaders/island_jump.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("island_parents", whole(&buffers.islands.parents)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        params: &FrameParams,
    ) {
        let constraint_active = params.constraint_count > 0;
        let archived = buffers.archive_capacity();
        let contacts = buffers.contact_capacity();

        self.contact_relay.record_stride(recorder, archived);
        self.contact_begin.record_stride(recorder, contacts);
        self.island_init.record(recorder, params.dynamic_count);
        self.island_link_contacts.record_stride(recorder, contacts);
        if constraint_active {
            self.island_link_constraints
                .record(recorder, params.constraint_count);
        }
        for _ in 0..params.island_rounds {
            self.island_jump.record(recorder, params.dynamic_count);
        }
    }
}
