use super::Count;
use super::Frame;
use super::buffers::{RigidBuffers, StreamId};
use super::shader;
use super::shader::CORE;
use crate::dynamics::engine::Stage;
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
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("body_descs", StreamId::BodyDescriptors.whole()),
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
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("ccd_factor", StreamId::CcdFactor.whole()),
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
                buffers,
                &[
                    ("params", StreamId::Params.whole()),
                    ("body_states", StreamId::BodyStates.whole()),
                    ("body_descs", StreamId::BodyDescriptors.whole()),
                    ("colliders", StreamId::Colliders.whole()),
                    ("collider_owners", StreamId::ColliderOwners.whole()),
                    ("aabbs", StreamId::ColliderAabbs.whole()),
                    ("counters", StreamId::Counters.whole()),
                ],
                &RigidBuffers::shape_resources(),
            ),
        }
    }

    pub(super) fn record_advance(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        self.advance
            .record_rows(recorder, buffers, Count::Dynamic.rows(&frame.params));
    }

    pub(super) fn record_broadphase(
        &self,
        recorder: &mut ComputeRecorder,
        buffers: &RigidBuffers,
        frame: &Frame,
    ) {
        self.broadphase_aabb
            .record_rows(recorder, buffers, Count::Colliders.rows(&frame.params));
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
                StreamId::JointFilterMajor.whole(),
                StreamId::JointFilterMinor.whole(),
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
            .record_rows(recorder, buffers, Count::Dynamic.rows(&frame.params));
        self.broadphase_aabb
            .record_rows(recorder, buffers, Count::Colliders.rows(&frame.params));
    }
}
