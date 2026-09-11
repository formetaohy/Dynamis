use super::FrameParams;
use super::dispatch::{CONTACT_BEGIN, CONTACT_RELAY, ISLAND_LINK_CONTACTS};
use super::stage::{CONTACT, CORE, RO, RW, Stage, UNIFORM, whole};
use crate::buffers::WorldBuffers;
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
                include_str!("../shaders/contact_relay.wgsl"),
                per_row,
                CONTACT,
                &[
                    (RO, whole(&buffers.contacts.archive)),
                    (RW, buffers.counter(COUNTER_ARCHIVED)),
                    (RW, whole(&buffers.contacts.manifolds)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RW, whole(&buffers.contacts.contact_matched)),
                    (RW, whole(&buffers.events)),
                    (RW, buffers.counter(COUNTER_EVENTS)),
                    (RW, buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.bodies.rows)),
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                ],
                &[],
            ),
            contact_begin: Stage::build(
                context,
                "contact_begin",
                include_str!("../shaders/contact_begin.wgsl"),
                per_row,
                CONTACT,
                &[
                    (RW, whole(&buffers.contacts.manifolds)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RO, whole(&buffers.contacts.contact_matched)),
                    (RW, whole(&buffers.events)),
                    (RW, buffers.counter(COUNTER_EVENTS)),
                    (RW, buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.contacts.resting)),
                    (RO, whole(&buffers.contacts.resting_live)),
                    (RO, whole(&buffers.contacts.resting_index.major)),
                    (RO, whole(&buffers.contacts.resting_index.minor)),
                    (RO, whole(&buffers.contacts.resting_index.payload)),
                    (RW, buffers.counter(COUNTER_RESTING_INDEX)),
                ],
                &[],
            ),
            island_init: Stage::build(
                context,
                "island_init",
                include_str!("../shaders/island_init.wgsl"),
                per_row,
                CORE,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.islands.parents)),
                    (RW, whole(&buffers.islands.state)),
                ],
                &[],
            ),
            island_link_contacts: Stage::build(
                context,
                "island_link_contacts",
                include_str!("../shaders/island_link_contacts.wgsl"),
                per_row,
                CORE,
                &[
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RO, whole(&buffers.contacts.manifolds)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RW, whole(&buffers.islands.parents)),
                    (RW, whole(&buffers.islands.wake_flags)),
                ],
                &[],
            ),
            island_link_constraints: Stage::build(
                context,
                "island_link_constraints",
                include_str!("../shaders/island_link_constraints.wgsl"),
                per_row,
                CORE,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                    (RO, whole(&buffers.constraints.descriptors)),
                    (RO, whole(&buffers.constraints.runtime)),
                    (RW, whole(&buffers.islands.parents)),
                    (RW, whole(&buffers.islands.wake_flags)),
                ],
                &[],
            ),
            island_jump: Stage::build(
                context,
                "island_jump",
                include_str!("../shaders/island_jump.wgsl"),
                per_row,
                CORE,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.islands.parents)),
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

        self.contact_relay
            .record_indirect(recorder, &buffers.dispatch, CONTACT_RELAY);
        self.contact_begin
            .record_indirect(recorder, &buffers.dispatch, CONTACT_BEGIN);
        self.island_init.record(recorder, params.dynamic_count);
        self.island_link_contacts.record_indirect(
            recorder,
            &buffers.dispatch,
            ISLAND_LINK_CONTACTS,
        );
        if constraint_active {
            self.island_link_constraints
                .record(recorder, params.constraint_count);
        }
        for _ in 0..params.island_rounds {
            self.island_jump.record(recorder, params.dynamic_count);
        }
    }
}
