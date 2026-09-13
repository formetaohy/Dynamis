use super::streams::RigidStream;
use crate::sort;
use dynamis_abi::COUNTER_JOINTS;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_kernels::{CORE, rows};
use dynamis_pass::Resources;
use dynamis_pass::Stage;
use dynamis_sort::RadixSort;
use dynamis_state::Count;
use dynamis_state::StateStream;
use dynamis_state::StepFrame;

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
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
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
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
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
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("colliders", StateStream::Colliders.whole()),
                    ("collider_owners", StateStream::ColliderOwners.whole()),
                    ("aabbs", RigidStream::ColliderAabbs.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &dynamis_state::shape_resources(),
            ),
        }
    }

    pub fn record_advance(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &StepFrame,
    ) {
        self.advance
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
    }

    pub fn record_broadphase(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &StepFrame,
    ) {
        self.broadphase_aabb
            .record_rows(recorder, streams, Count::Colliders.rows(&frame.params));
    }

    pub fn record(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &StepFrame,
        sort: &RadixSort,
    ) {
        if frame.params.constraint_count > 0 {
            let words = frame.shape.body_words;
            let channels = sort::lanes_dual(
                streams,
                dynamis_state::counter(COUNTER_JOINTS),
                RigidStream::JointFilterMajor.whole(),
                RigidStream::JointFilterMinor.whole(),
            );
            sort.sort(recorder, &channels, words, words);
        }
        self.integrate
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        self.broadphase_aabb
            .record_rows(recorder, streams, Count::Colliders.rows(&frame.params));
    }
}
