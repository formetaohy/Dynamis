use super::streams::RigidStream;
use super::{CONSTRAINT_LINK, CONTACT, CONTACT_ROW, IDENTITY_LINK, IDENTITY_LINK_ROW};
use crate::RigidFrame;
use dynamis_abi::Count;
use dynamis_abi::{
    COUNTER_ARCHIVED, COUNTER_CONTACTS, COUNTER_EVENTS, COUNTER_REFUSED_EVENTS, COUNTER_RESTING,
    COUNTER_RESTING_INDEX, COUNTER_SLEPT, COUNTER_WOKE, COUNTER_WOKE_DEFERRED,
};
use dynamis_gpu::ResourceSource;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{PassRuntime, Stage};
use dynamis_shader::{CORE, rows, stream};
use dynamis_state::StateStream;

pub struct BuildIslands {
    contact_relay: Stage,
    contact_begin: Stage,
    island_init: Stage,
    island_link_contacts: Stage,
    island_link_constraints: Stage,
    island_link_resting: Stage,
    island_jump: Stage,
    island_aggregate: Stage,
}

pub struct Wake {
    island_wake: Stage,
}

pub struct Sleep {
    island_sleep: Stage,
}

impl PassRuntime<RigidFrame> for BuildIslands {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            contact_relay: Stage::build(
                context,
                "contact_relay",
                stream(
                    context,
                    include_str!("../shaders/contact_relay.wgsl"),
                    CONTACT_ROW,
                    "work",
                    RigidStream::ContactArchive,
                ),
                streams,
                &[
                    ("archive", RigidStream::ContactArchive.whole()),
                    ("archive_count", dynamis_state::counter(COUNTER_ARCHIVED)),
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
                    ("contact_matched", RigidStream::ContactMatched.whole()),
                    ("events", RigidStream::Events.whole()),
                    ("event_count", dynamis_state::counter(COUNTER_EVENTS)),
                    ("spillover", dynamis_state::counter(COUNTER_REFUSED_EVENTS)),
                    ("params", StateStream::Params.whole()),
                    ("row_of_body", StateStream::BodyRowOfId.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
            contact_begin: Stage::build(
                context,
                "contact_begin",
                stream(
                    context,
                    include_str!("../shaders/contact_begin.wgsl"),
                    CONTACT,
                    "work",
                    RigidStream::Contacts,
                ),
                streams,
                &[
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
                    ("contact_matched", RigidStream::ContactMatched.whole()),
                    ("events", RigidStream::Events.whole()),
                    ("event_count", dynamis_state::counter(COUNTER_EVENTS)),
                    ("spillover", dynamis_state::counter(COUNTER_REFUSED_EVENTS)),
                    ("params", StateStream::Params.whole()),
                    ("resting", RigidStream::RestingContacts.whole()),
                    ("resting_live", RigidStream::RestingLive.whole()),
                    ("resting_major", RigidStream::RestingIndexMajor.whole()),
                    ("resting_minor", RigidStream::RestingIndexMinor.whole()),
                    ("resting_slots", RigidStream::RestingIndexSlots.whole()),
                    (
                        "resting_index",
                        dynamis_state::counter(COUNTER_RESTING_INDEX),
                    ),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
            island_init: Stage::build(
                context,
                "island_init",
                rows(
                    context,
                    include_str!("../shaders/island_init.wgsl"),
                    CORE,
                    Count::Dynamic.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("island_state", RigidStream::IslandState.whole()),
                ],
                &[],
            ),
            island_link_contacts: Stage::build(
                context,
                "island_link_contacts",
                stream(
                    context,
                    include_str!("../shaders/island_link_contacts.wgsl"),
                    IDENTITY_LINK,
                    "work",
                    RigidStream::Contacts,
                ),
                streams,
                &[
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_count", dynamis_state::counter(COUNTER_CONTACTS)),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                    ("params", StateStream::Params.whole()),
                    ("collider_owners", StateStream::ColliderOwners.whole()),
                ],
                &[],
            ),
            island_link_constraints: Stage::build(
                context,
                "island_link_constraints",
                rows(
                    context,
                    include_str!("../shaders/island_link_constraints.wgsl"),
                    CONSTRAINT_LINK,
                    Count::Constraints.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("constraint_runtime", StateStream::ConstraintRuntime.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                    ("constraint_rows", RigidStream::ConstraintRows.whole()),
                ],
                &[],
            ),
            island_link_resting: Stage::build(
                context,
                "island_link_resting",
                stream(
                    context,
                    include_str!("../shaders/island_link_resting.wgsl"),
                    IDENTITY_LINK_ROW,
                    "work",
                    RigidStream::RestingContacts,
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("resting", RigidStream::RestingContacts.whole()),
                    ("resting_live", RigidStream::RestingLive.whole()),
                    ("resting_count", dynamis_state::counter(COUNTER_RESTING)),
                    ("row_of_body", StateStream::BodyRowOfId.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                ],
                &[],
            ),
            island_jump: Stage::build(
                context,
                "island_jump",
                rows(
                    context,
                    include_str!("../shaders/island_jump.wgsl"),
                    CORE,
                    Count::Dynamic.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                ],
                &[],
            ),
            island_aggregate: Stage::build(
                context,
                "island_aggregate",
                rows(
                    context,
                    include_str!("../shaders/island_aggregate.wgsl"),
                    CORE,
                    Count::Dynamic.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_motion", RigidStream::BodyMotion.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("island_state", RigidStream::IslandState.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        let dynamic = Count::Dynamic.rows(&frame.params, &frame.rows);
        let constraints = Count::Constraints.rows(&frame.params, &frame.rows);
        self.contact_relay.record_stream(recorder, streams);
        self.contact_begin.record_stream(recorder, streams);
        self.island_init.record_rows(recorder, streams, dynamic);
        self.island_link_contacts.record_stream(recorder, streams);
        self.island_link_constraints
            .record_rows(recorder, streams, constraints);
        self.island_link_resting.record_stream(recorder, streams);
        for _ in 0..frame.shape.island_rounds {
            self.island_jump.record_rows(recorder, streams, dynamic);
        }
        self.island_aggregate
            .record_rows(recorder, streams, dynamic);
    }
}

impl PassRuntime<RigidFrame> for Wake {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            island_wake: Stage::build(
                context,
                "island_wake",
                rows(
                    context,
                    include_str!("../shaders/island_wake.wgsl"),
                    CORE,
                    Count::Dynamic.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("island_state", RigidStream::IslandState.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                    ("woke_count", dynamis_state::counter(COUNTER_WOKE)),
                    (
                        "deferred_woke_count",
                        dynamis_state::counter(COUNTER_WOKE_DEFERRED),
                    ),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        self.island_wake.record_rows(
            recorder,
            streams,
            Count::Dynamic.rows(&frame.params, &frame.rows),
        );
    }
}

impl PassRuntime<RigidFrame> for Sleep {
    fn build(context: &GpuContext, streams: &impl ResourceSource) -> Self {
        Self {
            island_sleep: Stage::build(
                context,
                "island_sleep",
                rows(
                    context,
                    include_str!("../shaders/island_sleep.wgsl"),
                    CORE,
                    Count::Dynamic.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("island_state", RigidStream::IslandState.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                    ("slept_count", dynamis_state::counter(COUNTER_SLEPT)),
                ],
                &[],
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) {
        self.island_sleep.record_rows(
            recorder,
            streams,
            Count::Dynamic.rows(&frame.params, &frame.rows),
        );
    }
}
