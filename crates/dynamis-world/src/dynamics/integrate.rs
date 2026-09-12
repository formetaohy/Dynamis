use super::FrameParams;
use super::stage::{CORE, Stage, shape_resources, whole};
use crate::dynamics::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_JOINTS;
use dynamis_sort::RadixSort;

pub(super) struct Integrate {
    integrate: Stage,
    advance: Stage,
    broadphase_aabb: Stage,
}

impl Integrate {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            integrate: Stage::build(
                context,
                "integrate",
                include_str!("shaders/integrate.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.bodies.states)),
                    ("body_descs", whole(&buffers.bodies.descriptors)),
                ],
                &[],
            ),
            advance: Stage::build(
                context,
                "advance",
                include_str!("shaders/advance.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.bodies.states)),
                ],
                &[],
            ),
            broadphase_aabb: Stage::build(
                context,
                "broadphase_aabb",
                include_str!("shaders/broadphase_aabb.wgsl"),
                per_row,
                CORE,
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.bodies.states)),
                    ("colliders", whole(&buffers.bodies.colliders)),
                    ("aabbs", whole(&buffers.bodies.aabbs)),
                ],
                &shape_resources(buffers),
            ),
        }
    }

    pub(super) fn record_advance(&self, recorder: &mut ComputeRecorder, dynamic_count: u32) {
        self.advance.record(recorder, dynamic_count);
    }

    pub(super) fn record_broadphase(&self, recorder: &mut ComputeRecorder, dynamic_count: u32) {
        self.broadphase_aabb.record(recorder, dynamic_count);
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        params: &FrameParams,
        sort: &RadixSort,
    ) {
        if params.constraint_count > 0 {
            let words = buffers.body_words();
            let channels = buffers.sort_lanes_dual(
                buffers.counter(COUNTER_JOINTS),
                &buffers.constraints.joint_major,
                &buffers.constraints.joint_minor,
            );
            sort.sort(
                recorder,
                &channels,
                words,
                words,
                buffers.constraint_capacity(),
            );
        }
        self.integrate.record(recorder, params.dynamic_count);
        self.broadphase_aabb.record(recorder, params.dynamic_count);
    }
}
