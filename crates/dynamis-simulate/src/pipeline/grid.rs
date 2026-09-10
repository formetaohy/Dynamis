use super::stage::{RO, RW, Stage, UNIFORM, whole};
use crate::buffers::WorldBuffers;
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
                include_str!("../shaders/grid_entries.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.bodies.aabbs)),
                    (RW, whole(&buffers.contacts.entries.cells)),
                    (RW, whole(&buffers.contacts.entries.colliders)),
                    (RW, buffers.counter(COUNTER_ENTRIES)),
                    (RW, whole(&buffers.contacts.large_bodies)),
                    (RW, buffers.counter(COUNTER_LARGE)),
                    (RW, buffers.counter(COUNTER_SPILLOVER_ENTRIES)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(&self, recorder: &mut ComputeRecorder, body_count: u32) {
        self.grid_entries.record(recorder, body_count);
    }
}
