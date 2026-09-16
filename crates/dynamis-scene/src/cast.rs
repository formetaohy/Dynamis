use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::{ComputeRecorder, GpuContext, Resources, SlotRef};
use dynamis_pass::Stage;
use dynamis_shader::{GEOMETRY_INDEX, SCENE_CAST, workgroups};
use dynamis_soft::SoftStream;
use dynamis_state::StateStream;

pub struct SceneCast {
    stage: Stage,
}

impl SceneCast {
    pub fn build(
        context: &GpuContext,
        streams: &impl Resources,
        queries: SlotRef,
        results: SlotRef,
    ) -> Self {
        let slots = [
            ("queries", queries),
            ("body_states", StateStream::BodyStates.whole()),
            ("body_descs", StateStream::BodyDescriptors.whole()),
            ("colliders", StateStream::Colliders.whole()),
            ("entry_keys", BroadphaseStream::EntryKeys.whole()),
            ("entry_order", BroadphaseStream::EntryOrder.whole()),
            ("entries", BroadphaseStream::Entries.whole()),
            ("counters", StateStream::Counters.whole()),
            ("query_results", results),
            ("params", StateStream::Params.whole()),
            ("collider_owners", StateStream::ColliderOwners.whole()),
            ("particles", SoftStream::Particles.whole()),
            ("soft_bodies", SoftStream::BodyStates.whole()),
        ];
        Self {
            stage: Stage::build(
                context,
                "scene cast",
                workgroups(context, SCENE_CAST, GEOMETRY_INDEX),
                streams,
                &slots,
                &dynamis_state::shape_resources(),
            ),
        }
    }

    pub fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        workgroups: u32,
    ) {
        self.stage.record_workgroups(recorder, streams, workgroups);
    }
}
