use super::stage::{CORE, Stage, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::{COUNTER_ENTRIES, COUNTER_LARGE, COUNTER_SPILLOVER_ENTRIES};

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
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("aabbs", whole(&buffers.bodies.aabbs)),
                    ("entry_cells", whole(&buffers.contacts.entries.cells)),
                    (
                        "entry_colliders",
                        whole(&buffers.contacts.entries.colliders),
                    ),
                    ("entry_count", buffers.counter(COUNTER_ENTRIES)),
                    ("large_bodies", whole(&buffers.contacts.large_bodies)),
                    ("large_count", buffers.counter(COUNTER_LARGE)),
                    ("spillover", buffers.counter(COUNTER_SPILLOVER_ENTRIES)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, body_count: u32) {
        self.grid_entries.record(recorder, body_count);
    }
}
