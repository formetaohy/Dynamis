use super::streams::RigidStream;
use crate::RigidFrame;
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::Resources;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{PassRuntime, Stage};
use dynamis_shader::GEOMETRY_INDEX;
use dynamis_shader::workgroups;
use dynamis_state::StateStream;

pub struct Query {
    kernel: Stage,
}

impl PassRuntime<RigidFrame> for Query {
    fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            kernel: Stage::build(
                context,
                "query",
                workgroups(
                    context,
                    include_str!("../shaders/queries.wgsl"),
                    GEOMETRY_INDEX,
                ),
                streams,
                &[
                    ("queries", StateStream::QueryRecords.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("colliders", StateStream::Colliders.whole()),
                    ("aabbs", RigidStream::ColliderAabbs.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                    ("query_results", StateStream::QueryResults.whole()),
                    ("params", StateStream::Params.whole()),
                    ("collider_owners", StateStream::ColliderOwners.whole()),
                ],
                &dynamis_state::shape_resources(),
            ),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.kernel
            .record_workgroups(recorder, streams, frame.query_count);
    }
}
