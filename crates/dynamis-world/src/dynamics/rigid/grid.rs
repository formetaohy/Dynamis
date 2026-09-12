use super::Count;
use super::shader;
use super::shader::GRID_INDEX;
use super::streams::RigidStream;
use crate::dynamics::Frame;
use crate::dynamics::engine::Stage;
use crate::dynamics::scene::SceneStream;
use crate::dynamics::streams::Streams;
use dynamis_gpu::{ComputeRecorder, GpuContext};

pub(super) struct Grid {
    grid_entries: Stage,
}

impl Grid {
    pub(super) fn build(context: &GpuContext, streams: &Streams) -> Self {
        Self {
            grid_entries: Stage::build(
                context,
                "grid_entries",
                shader::rows(
                    context,
                    include_str!("shaders/grid_entries.wgsl"),
                    GRID_INDEX,
                    Count::Colliders,
                ),
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("aabbs", RigidStream::ColliderAabbs.whole()),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
                    ("body_activity", RigidStream::BodyActivity.whole()),
                    ("entry_keys", RigidStream::GridEntryKeys.whole()),
                    ("entry_colliders", RigidStream::GridEntryColliders.whole()),
                    ("counters", SceneStream::Counters.whole()),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, streams: &Streams, frame: &Frame) {
        self.grid_entries
            .record_rows(recorder, streams, Count::Colliders.rows(&frame.params));
    }
}
