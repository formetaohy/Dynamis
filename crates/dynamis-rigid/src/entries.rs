use super::streams::RigidStream;
use dynamis_broadphase::BroadphaseStream;
use dynamis_engine::Resources;
use dynamis_engine::Stage;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernels::{GRID_INDEX, rows};
use dynamis_scene::Count;
use dynamis_scene::Frame;
use dynamis_scene::SceneStream;

pub struct Entries {
    emit: Stage,
}

impl Entries {
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            emit: Stage::build(
                context,
                "collider_entries",
                rows(
                    context,
                    include_str!("../shaders/collider_entries.wgsl"),
                    GRID_INDEX,
                    Count::Colliders.field(),
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("aabbs", RigidStream::ColliderAabbs.whole()),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
                    ("body_activity", RigidStream::BodyActivity.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", SceneStream::Counters.whole()),
                ],
                &[],
            ),
        }
    }

    pub fn record(&self, recorder: &mut ComputeRecorder, streams: &impl Resources, frame: &Frame) {
        self.emit
            .record_rows(recorder, streams, Count::Colliders.rows(&frame.params));
    }
}
