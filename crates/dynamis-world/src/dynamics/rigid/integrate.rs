use super::Count;
use super::Frame;
use super::buffers::RigidBuffers;
use super::shader;
use super::shader::CORE;
use crate::dynamics::engine::{Stage, whole};
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_JOINTS;
use dynamis_sort::RadixSort;

pub(super) struct Integrate {
    integrate: Stage,
    advance: Stage,
    broadphase_aabb: Stage,
}

impl Integrate {
    pub(super) fn build(context: &GpuContext, buffers: &RigidBuffers) -> Self {
        Self {
            integrate: Stage::build(
                context,
                "integrate",
                shader::rows(
                    context,
                    include_str!("shaders/integrate.wgsl"),
                    CORE,
                    Count::Dynamic,
                ),
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
                shader::rows(
                    context,
                    include_str!("shaders/advance.wgsl"),
                    CORE,
                    Count::Dynamic,
                ),
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
                shader::rows(
                    context,
                    include_str!("shaders/broadphase_aabb.wgsl"),
                    CORE,
                    Count::Colliders,
                ),
                &[
                    ("params", whole(&buffers.params)),
                    ("body_states", whole(&buffers.body_states)),
                    ("body_descs", whole(&buffers.body_descriptors)),
                    ("colliders", whole(&buffers.colliders)),
                    ("collider_owners", whole(&buffers.collider_owners)),
                    ("aabbs", whole(&buffers.collider_aabbs)),
                    ("counters", whole(&buffers.counters)),
                ],
                &buffers.shape_resources(),
            ),
        }
    }

    pub(super) fn record_advance(&self, recorder: &mut ComputeRecorder, frame: &Frame) {
        self.advance
            .record_rows(recorder, Count::Dynamic.rows(&frame.params));
    }

    pub(super) fn record_broadphase(&self, recorder: &mut ComputeRecorder, frame: &Frame) {
        self.broadphase_aabb
            .record_rows(recorder, Count::Colliders.rows(&frame.params));
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
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
        self.integrate
            .record_rows(recorder, Count::Dynamic.rows(&frame.params));
        self.broadphase_aabb
            .record_rows(recorder, Count::Colliders.rows(&frame.params));
    }
}
