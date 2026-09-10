use super::FrameParams;
use super::dispatch::SORT_JOINTS;
use super::sort;
use super::stage::{RO, RW, Stage, UNIFORM, shape_resources, whole};
use crate::buffers::WorldBuffers;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_JOINTS;
use dynamis_sort::{RadixSort, key_words};

pub(super) struct Integrate {
    integrate: Stage,
    broadphase_aabb: Stage,
}

impl Integrate {
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers, per_row: u32) -> Self {
        Self {
            integrate: Stage::build(
                context,
                "integrate",
                include_str!("../shaders/integrate.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RW, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.descriptors)),
                ],
                &[],
            ),
            broadphase_aabb: Stage::build(
                context,
                "broadphase_aabb",
                include_str!("../shaders/broadphase_aabb.wgsl"),
                per_row,
                &[
                    (UNIFORM, whole(&buffers.params)),
                    (RO, whole(&buffers.bodies.states)),
                    (RO, whole(&buffers.bodies.colliders)),
                    (RW, whole(&buffers.bodies.aabbs)),
                ],
                &shape_resources(buffers),
            ),
        }
    }

    /// Rebuilds the broadphase scene a query runs against, without integrating.
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
            let words = key_words(buffers.constraint_rows().max(1));
            let channels = sort::lanes(
                buffers,
                buffers.counter(COUNTER_JOINTS),
                &buffers.constraints.joint_lo,
                &buffers.constraints.joint_hi,
                &buffers.sort.values,
            );
            sort.sort(
                recorder,
                &channels,
                words,
                words,
                &buffers.dispatch,
                SORT_JOINTS,
            );
        }
        self.integrate.record(recorder, params.dynamic_count);
        self.broadphase_aabb.record(recorder, params.dynamic_count);
    }
}
