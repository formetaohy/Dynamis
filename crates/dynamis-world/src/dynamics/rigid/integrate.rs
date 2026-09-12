use super::Count;
use super::shader;
use super::shader::CORE;
use super::streams::RigidStream;
use crate::dynamics::Frame;
use crate::dynamics::engine::Stage;
use crate::dynamics::scene::SceneStream;
use crate::dynamics::streams::Streams;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_layout::COUNTER_JOINTS;
use dynamis_sort::RadixSort;

pub(super) struct Integrate {
    integrate: Stage,
    advance: Stage,
    broadphase_aabb: Stage,
}

impl Integrate {
    pub(super) fn build(context: &GpuContext, streams: &Streams) -> Self {
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
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
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
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("ccd_factor", RigidStream::CcdFactor.whole()),
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
                streams,
                &[
                    ("params", SceneStream::Params.whole()),
                    ("body_states", SceneStream::BodyStates.whole()),
                    ("body_descs", SceneStream::BodyDescriptors.whole()),
                    ("colliders", SceneStream::Colliders.whole()),
                    ("collider_owners", SceneStream::ColliderOwners.whole()),
                    ("aabbs", RigidStream::ColliderAabbs.whole()),
                    ("counters", SceneStream::Counters.whole()),
                ],
                &streams.scene.shape_resources(),
            ),
        }
    }

    pub(super) fn record_advance(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &Streams,
        frame: &Frame,
    ) {
        self.advance
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
    }

    pub(super) fn record_broadphase(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &Streams,
        frame: &Frame,
    ) {
        self.broadphase_aabb
            .record_rows(recorder, streams, Count::Colliders.rows(&frame.params));
    }

    pub(super) fn record(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &Streams,
        frame: &Frame,
        sort: &RadixSort,
    ) {
        if frame.params.constraint_count > 0 {
            let words = streams.scene.body_words();
            let channels = streams.sort_lanes_dual(
                streams.scene.counter(COUNTER_JOINTS),
                RigidStream::JointFilterMajor.whole(),
                RigidStream::JointFilterMinor.whole(),
            );
            sort.sort(
                recorder,
                &channels,
                words,
                words,
                streams.scene.constraint_capacity(),
            );
        }
        self.integrate
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        self.broadphase_aabb
            .record_rows(recorder, streams, Count::Colliders.rows(&frame.params));
    }
}
