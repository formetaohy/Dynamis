use super::streams::RigidStream;
use crate::RigidFrame;
use dynamis_abi::Count;
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::Resources;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::Stage;
use dynamis_shader::{GRID_INDEX, rows};
use dynamis_state::StateStream;

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
                    Count::Colliders.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("aabbs", RigidStream::ColliderAabbs.whole()),
                    ("collider_owners", StateStream::ColliderOwners.whole()),
                    ("body_activity", RigidStream::BodyActivity.whole()),
                    ("wake_flags", StateStream::WakeFlags.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
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
        &mut self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.emit.record_rows(
            recorder,
            streams,
            Count::Colliders.rows(&frame.params, &frame.rows),
        );
    }
}
