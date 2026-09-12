use super::stage::{GRID_INDEX, Stage, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{
    COUNTER_COARSE_ACTIVE, COUNTER_ENTRIES, COUNTER_GRID_LEVELS, COUNTER_SPILLOVER_ENTRIES,
};

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
                    ("aabbs", whole(&buffers.bodies.aabbs)),
                    ("collider_owners", whole(&buffers.bodies.collider_owners)),
                    ("body_activity", whole(&buffers.bodies.activity)),
                    ("entry_keys", whole(&buffers.contacts.entries.keys)),
                    (
                        "entry_colliders",
                        whole(&buffers.contacts.entries.colliders),
                    ),
                    ("entry_count", buffers.counter(COUNTER_ENTRIES)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_ENTRIES)),
                    ("levels", buffers.counter(COUNTER_GRID_LEVELS)),
                    ("active_coarse", buffers.counter(COUNTER_COARSE_ACTIVE)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, collider_count: u32) {
        self.grid_entries.record(recorder, collider_count);
    }
}
