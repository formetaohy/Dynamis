use super::stage::{CORE, GEOMETRY, Stage, shape_resources, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_PAIRS;

pub(super) struct Ccd {
    sweep: Stage,
    apply: Stage,
}

impl Ccd {
    pub(super) fn build(
        context: &GpuContext,
        buffers: &WorldBuffers,
        max_workgroups_per_dimension: u32,
    ) -> Self {
        Self {
            sweep: Stage::build(
                context,
                "ccd_sweep",
                include_str!("shaders/ccd_sweep.wgsl"),
                max_workgroups_per_dimension,
                GEOMETRY,
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
                max_workgroups_per_dimension,
                CORE,
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
        dynamic_count: u32,
    ) {
        self.sweep.record_stride(recorder, buffers.pair_capacity());
        self.apply.record(recorder, dynamic_count);
    }
}
