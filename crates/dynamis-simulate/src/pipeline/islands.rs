use super::FrameParams;
use super::dispatch::{
    CONTACT_MATCH, GATHER_CONTACT_KEYS_B, ISLAND_LINK_CONTACTS, MARK_CONTACT_BOUNDARIES,
    SORT_CONSTRAINTS, SORT_CONTACTS,
};
use super::stage::{CORE, RO, RW, Stage, UNIFORM, whole};
use crate::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{
    COUNTER_CONSTRAINTS, COUNTER_CONTACTS, COUNTER_EVENTS, COUNTER_PREV_CONTACTS,
    COUNTER_SPILLOVER_EVENTS,
};
use dynamis_sort::RadixSort;

pub(super) struct Islands {
    contact_match: Stage,
    island_init: Stage,
    island_link_contacts: Stage,
    island_link_constraints: Stage,
    island_jump: Stage,
    gather_contact_keys_b: Stage,
    gather_constraint_keys: Stage,
    reset_gather_boundaries: Stage,
    mark_contact_boundaries: Stage,
    mark_constraint_boundaries: Stage,
}

impl Islands {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            contact_match: Stage::build(
                context,
                "contact_match",
                include_str!("../shaders/contact_match.wgsl"),
                per_row,
                CORE,
                &[
                    (RW, whole(&buffers.contacts.manifolds)),
                    (RO, whole(&buffers.contacts.previous)),
                    (RW, buffers.counter(COUNTER_PREV_CONTACTS)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RW, whole(&buffers.events)),
                    (RW, buffers.counter(COUNTER_EVENTS)),
                    (RW, buffers.counter(COUNTER_SPILLOVER_EVENTS)),
                    (UNIFORM, whole(&buffers.params)),
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
            gather_contact_keys_b: Stage::build(
                context,
                "gather_contact_keys_b",
                include_str!("../shaders/gather_contact_keys_b.wgsl"),
                per_row,
                CORE,
                &[
                    (RO, whole(&buffers.contacts.manifolds)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RW, whole(&buffers.contacts.b_bodies)),
                    (RW, whole(&buffers.contacts.b_rows)),
                ],
                &[],
            ),
            gather_constraint_keys: Stage::build(
                context,
                "gather_constraint_keys",
                include_str!("../shaders/gather_constraint_keys.wgsl"),
                per_row,
                CORE,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.constraints.runtime)),
                    (RW, whole(&buffers.constraints.a_bodies)),
                    (RW, whole(&buffers.constraints.a_rows)),
                    (RW, whole(&buffers.constraints.b_bodies)),
                    (RW, whole(&buffers.constraints.b_rows)),
                    (RO, whole(&buffers.constraints.descriptors)),
                ],
                &[],
            ),
            reset_gather_boundaries: Stage::build(
                context,
                "reset_gather_boundaries",
                include_str!("../shaders/reset_gather_boundaries.wgsl"),
                per_row,
                CORE,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.contacts.first_a)),
                    (RW, whole(&buffers.contacts.first_b)),
                    (RW, whole(&buffers.constraints.first_a)),
                    (RW, whole(&buffers.constraints.first_b)),
                ],
                &[],
            ),
            mark_contact_boundaries: Stage::build(
                context,
                "mark_contact_boundaries",
                include_str!("../shaders/mark_contact_boundaries.wgsl"),
                per_row,
                CORE,
                &[
                    (RO, whole(&buffers.contacts.a_body)),
                    (RO, whole(&buffers.contacts.b_bodies)),
                    (RW, buffers.counter(COUNTER_CONTACTS)),
                    (RW, whole(&buffers.contacts.first_a)),
                    (RW, whole(&buffers.contacts.first_b)),
                ],
                &[],
            ),
            mark_constraint_boundaries: Stage::build(
                context,
                "mark_constraint_boundaries",
                include_str!("../shaders/mark_constraint_boundaries.wgsl"),
                per_row,
                CORE,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.constraints.a_bodies)),
                    (RO, whole(&buffers.constraints.b_bodies)),
                    (RW, whole(&buffers.constraints.first_a)),
                    (RW, whole(&buffers.constraints.first_b)),
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
        sort: &RadixSort,
    ) {
        let constraint_active = params.constraint_count > 0;
        let body_words = buffers.body_words();

        self.contact_match
            .record_indirect(recorder, &buffers.dispatch, CONTACT_MATCH);
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
        self.gather_contact_keys_b.record_indirect(
            recorder,
            &buffers.dispatch,
            GATHER_CONTACT_KEYS_B,
        );
        let channels = buffers.sort_lanes(
            buffers.counter(COUNTER_CONTACTS),
            &buffers.contacts.b_bodies,
            &buffers.contacts.b_rows,
        );
        sort.sort(
            recorder,
            &channels,
            body_words,
            0,
            &buffers.dispatch,
            SORT_CONTACTS,
        );
        if constraint_active {
            self.gather_constraint_keys
                .record(recorder, params.constraint_count);
            for (bodies, rows) in [
                (&buffers.constraints.a_bodies, &buffers.constraints.a_rows),
                (&buffers.constraints.b_bodies, &buffers.constraints.b_rows),
            ] {
                let channels =
                    buffers.sort_lanes(buffers.counter(COUNTER_CONSTRAINTS), bodies, rows);
                sort.sort(
                    recorder,
                    &channels,
                    body_words,
                    0,
                    &buffers.dispatch,
                    SORT_CONSTRAINTS,
                );
            }
        }
        self.reset_gather_boundaries
            .record(recorder, params.dynamic_count);
        self.mark_contact_boundaries.record_indirect(
            recorder,
            &buffers.dispatch,
            MARK_CONTACT_BOUNDARIES,
        );
        if constraint_active {
            self.mark_constraint_boundaries
                .record(recorder, params.constraint_count);
        }
    }
}
