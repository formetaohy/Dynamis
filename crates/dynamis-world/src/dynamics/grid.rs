use super::stage::{GRID_INDEX, Stage, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};

pub(super) struct Grid {
    grid_entries: Stage,
}

impl Grid {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            grid_entries: Stage::build(
                context,
                "grid_entries",
                include_str!("shaders/grid_entries.wgsl"),
                per_row,
                GRID_INDEX,
                &[
                    ("params", whole(&buffers.params)),
                    ("aabbs", whole(&buffers.collider_aabbs)),
                    ("collider_owners", whole(&buffers.collider_owners)),
                    ("body_activity", whole(&buffers.body_activity)),
                    ("entry_keys", whole(&buffers.grid_entry_keys)),
                    ("entry_colliders", whole(&buffers.grid_entry_colliders)),
                    ("counters", whole(&buffers.counters)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, collider_count: u32) {
        self.grid_entries.record(recorder, collider_count);
    }
}
