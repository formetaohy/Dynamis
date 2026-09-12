use super::Frame;
use super::stage::{CORE, Count, Coverage, Stage, shape_resources, whole};
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
    pub(super) fn build(context: &GpuContext, buffers: &WorldBuffers) -> Self {
        Self {
            integrate: Stage::build(
                context,
                "integrate",
                include_str!("shaders/integrate.wgsl"),
                CORE,
                Coverage::Live(Count::Dynamic),
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                ],
                &[],
            ),
            advance: Stage::build(
                context,
                "advance",
                include_str!("shaders/advance.wgsl"),
                CORE,
                Coverage::Live(Count::Dynamic),
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.body_states)),
                    ("ccd_factor", whole(&buffers.ccd_factor)),
                ],
                &[],
            ),
            broadphase_aabb: Stage::build(
                context,
                "broadphase_aabb",
                include_str!("shaders/broadphase_aabb.wgsl"),
                CORE,
                Coverage::Live(Count::Colliders),
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("colliders", whole(&buffers.colliders)),
                    ("collider_owners", whole(&buffers.collider_owners)),
                    ("aabbs", whole(&buffers.collider_aabbs)),
                    ("counters", whole(&buffers.counters)),
                ],
                &shape_resources(buffers),
            ),
        }
    }

    pub(super) fn record_advance(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        frame: &Frame,
    ) {
        self.advance.record(recorder, buffers, frame);
    }

    pub(super) fn record_broadphase(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        frame: &Frame,
    ) {
        self.broadphase_aabb.record(recorder, buffers, frame);
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &WorldBuffers,
        frame: &Frame,
        sort: &RadixSort,
    ) {
        if frame.params.constraint_count > 0 {
            let words = buffers.body_words();
            let channels = buffers.sort_lanes_dual(
                buffers.counter(COUNTER_JOINTS),
                &buffers.joint_filter_major,
                &buffers.joint_filter_minor,
            );
            sort.sort(
                recorder,
                &channels,
                words,
                words,
                buffers.constraint_capacity(),
            );
        }
        self.integrate.record(recorder, buffers, frame);
        self.broadphase_aabb.record(recorder, buffers, frame);
    }
}
