use super::streams::RigidStream;
use crate::sort;
use dynamis_abi::COUNTER_JOINTS;
use dynamis_engine::Resources;
use dynamis_engine::Stage;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernels::{CORE, rows};
use dynamis_scene::Count;
use dynamis_scene::Frame;
use dynamis_scene::SceneStream;
use dynamis_sort::RadixSort;

pub struct Integrate {
    integrate: Stage,
    advance: Stage,
    broadphase_aabb: Stage,
}

impl Integrate {
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            integrate: Stage::build(
                context,
                "integrate",
                rows(
                    context,
                    include_str!("../shaders/integrate.wgsl"),
                    CORE,
                    Count::Dynamic.field(),
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
                rows(
                    context,
                    include_str!("../shaders/advance.wgsl"),
                    CORE,
                    Count::Dynamic.field(),
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
                rows(
                    context,
                    include_str!("../shaders/broadphase_aabb.wgsl"),
                    CORE,
                    Count::Colliders.field(),
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
                &dynamis_scene::shape_resources(),
            ),
        }
    }

    pub fn record_advance(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &Frame,
    ) {
        self.advance
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
    }

    pub fn record_broadphase(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &Frame,
    ) {
        self.broadphase_aabb
            .record_rows(recorder, streams, Count::Colliders.rows(&frame.params));
    }

    pub fn record(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &Frame,
        sort: &RadixSort,
    ) {
        if frame.params.constraint_count > 0 {
            let words = dynamis_scene::body_words(streams);
            let channels = sort::lanes_dual(
                streams,
                dynamis_scene::counter(COUNTER_JOINTS),
                RigidStream::JointFilterMajor.whole(),
                RigidStream::JointFilterMinor.whole(),
            );
            sort.sort(
                recorder,
                &channels,
                words,
                words,
                dynamis_scene::constraint_capacity(streams),
            );
        }
        self.integrate
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        self.broadphase_aabb
            .record_rows(recorder, streams, Count::Colliders.rows(&frame.params));
    }
}
