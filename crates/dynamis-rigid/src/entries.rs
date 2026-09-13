use super::streams::RigidStream;
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernels::{GRID_INDEX, rows};
use dynamis_pass::Resources;
use dynamis_pass::Stage;
use dynamis_state::Count;
use dynamis_state::StateStream;
use dynamis_state::StepFrame;

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
                    ("params", StateStream::Params.whole()),
                    ("aabbs", RigidStream::ColliderAabbs.whole()),
                    ("collider_owners", StateStream::ColliderOwners.whole()),
                    ("body_activity", RigidStream::BodyActivity.whole()),
                    ("entry_keys", BroadphaseStream::EntryKeys.whole()),
                    ("entry_order", BroadphaseStream::EntryOrder.whole()),
                    ("entries", BroadphaseStream::Entries.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
        }
    }

    pub fn record(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &StepFrame,
    ) {
        self.emit
            .record_rows(recorder, streams, Count::Colliders.rows(&frame.params));
    }
}
