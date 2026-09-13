use super::streams::RigidStream;
use super::{CONTACT, IDENTITY};
use dynamis_abi::{
    COUNTER_ARCHIVED, COUNTER_CONTACTS, COUNTER_EVENTS, COUNTER_RESTING_INDEX,
    COUNTER_SPILLOVER_EVENTS,
};
use dynamis_engine::Resources;
use dynamis_engine::Stage;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernels::{CORE, rows, stream};
use dynamis_scene::Count;
use dynamis_scene::Frame;
use dynamis_scene::SceneStream;

pub struct Islands {
    contact_relay: Stage,
    contact_begin: Stage,
    island_init: Stage,
    island_link_contacts: Stage,
    island_link_constraints: Stage,
    island_jump: Stage,
}

impl Islands {
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            contact_relay: Stage::build(
                context,
                "contact_relay",
                stream(
                    context,
                    include_str!("../shaders/contact_relay.wgsl"),
                    CONTACT,
                    "work",
                    RigidStream::ContactArchive,
                ),
                streams,
                &[
                    ("archive", RigidStream::ContactArchive.whole()),
                    ("archive_count", dynamis_scene::counter(COUNTER_ARCHIVED)),
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_count", dynamis_scene::counter(COUNTER_CONTACTS)),
                    ("contact_matched", RigidStream::ContactMatched.whole()),
                    ("events", RigidStream::Events.whole()),
                    ("event_count", dynamis_scene::counter(COUNTER_EVENTS)),
                    (
                        "spillover",
                        dynamis_scene::counter(COUNTER_SPILLOVER_EVENTS),
                    ),
                    ("params", SceneStream::Params.whole()),
                    ("row_of_body", SceneStream::BodyRowOfId.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
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
                    ("contact_count", dynamis_scene::counter(COUNTER_CONTACTS)),
                    ("contact_matched", RigidStream::ContactMatched.whole()),
                    ("events", RigidStream::Events.whole()),
                    ("event_count", dynamis_scene::counter(COUNTER_EVENTS)),
                    (
                        "spillover",
                        dynamis_scene::counter(COUNTER_SPILLOVER_EVENTS),
                    ),
                    ("params", SceneStream::Params.whole()),
                    ("resting", RigidStream::RestingContacts.whole()),
                    ("resting_live", RigidStream::RestingLive.whole()),
                    ("resting_major", RigidStream::RestingIndexMajor.whole()),
                    ("resting_minor", RigidStream::RestingIndexMinor.whole()),
                    ("resting_slots", RigidStream::RestingIndexSlots.whole()),
                    (
                        "resting_index",
                        dynamis_scene::counter(COUNTER_RESTING_INDEX),
                    ),
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
                    Count::Dynamic.field(),
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
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
                    IDENTITY,
                    "work",
                    RigidStream::Contacts,
                ),
                streams,
                &[
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("contacts", RigidStream::Contacts.whole()),
                    ("contact_count", dynamis_scene::counter(COUNTER_CONTACTS)),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("wake_flags", SceneStream::WakeFlags.whole()),
                    ("params", SceneStream::Params.whole()),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
                ],
                &[],
            ),
            island_link_constraints: Stage::build(
                context,
                "island_link_constraints",
                rows(
                    context,
                    include_str!("../shaders/island_link_constraints.wgsl"),
                    CORE,
                    Count::Constraints.field(),
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("constraint_runtime", SceneStream::ConstraintRuntime.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                    ("wake_flags", SceneStream::WakeFlags.whole()),
                    ("constraint_rows", RigidStream::ConstraintRows.whole()),
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
                    Count::Dynamic.field(),
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("island_parents", RigidStream::IslandParents.whole()),
                ],
                &[],
            ),
        }
    }

    pub fn record(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &Frame,
        rounds: u32,
    ) {
        self.contact_relay.record_stream(recorder, streams);
        self.contact_begin.record_stream(recorder, streams);
        self.island_init
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        self.island_link_contacts.record_stream(recorder, streams);
        self.island_link_constraints.record_rows(
            recorder,
            streams,
            Count::Constraints.rows(&frame.params),
        );
        for _ in 0..rounds {
            self.island_jump
                .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        }
    }
}
