use super::Frame;
use super::stage::{CORE, Count, Coverage, GEOMETRY, Slots, Stage, shape_resources, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_PAIRS;

pub(super) struct Ccd {
    sweep: Stage,
    apply: Stage,
}

impl Ccd {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers) -> Self {
        Self {
            sweep: Stage::build(
                context,
                "ccd_sweep",
                include_str!("shaders/ccd_sweep.wgsl"),
                GEOMETRY,
                Coverage::Stream {
                    kernel: "work",
                    slots: Slots::Pairs,
                },
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("colliders", whole(&buffers.colliders)),
                    ("pair_major", whole(&buffers.pair_major)),
                    ("pair_minor", whole(&buffers.pair_minor)),
                    ("pair_count", buffers.counter(COUNTER_PAIRS)),
                    ("collider_owners", whole(&buffers.collider_owners)),
                    ("ccd_factor", whole(&buffers.ccd_factor)),
                    ("ccd_impact", whole(&buffers.ccd_impact)),
                ],
                &shape_resources(buffers),
            ),
            apply: Stage::build(
                context,
                "ccd_apply",
                include_str!("shaders/ccd_apply.wgsl"),
                CORE,
                Coverage::Live(Count::Dynamic),
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.body_states)),
                    ("ccd_factor", whole(&buffers.ccd_factor)),
                    ("ccd_impact", whole(&buffers.ccd_impact)),
                ],
                &[],
            ),
        }
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        frame: &Frame,
    ) {
        self.sweep.record(recorder, buffers, frame);
        self.apply.record(recorder, buffers, frame);
    }
}
