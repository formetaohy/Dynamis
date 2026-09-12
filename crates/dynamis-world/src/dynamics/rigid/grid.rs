use super::Count;
use super::Frame;
use super::buffers::{RigidBuffers, StreamId};
use super::shader;
use super::shader::GRID_INDEX;
use crate::dynamics::engine::Stage;
use dynamis_gpu::{ComputeRecorder, GpuContext};

pub(super) struct Grid {
    grid_entries: Stage,
}

impl Grid {
    pub(super) fn build(context: &GpuContext, buffers: &RigidBuffers) -> Self {
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
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("aabbs", StreamId::ColliderAabbs.whole()),
                    ("collider_owners", StreamId::ColliderOwners.whole()),
                    ("body_activity", StreamId::BodyActivity.whole()),
                    ("entry_keys", StreamId::GridEntryKeys.whole()),
                    ("entry_colliders", StreamId::GridEntryColliders.whole()),
                    ("counters", StreamId::Counters.whole()),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        self.grid_entries
            .record_rows(recorder, buffers, Count::Colliders.rows(&frame.params));
    }
}
