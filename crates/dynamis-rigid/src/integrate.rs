use super::streams::RigidStream;
use crate::RigidFrame;
use crate::sort;
use dynamis_abi::COUNTER_JOINTS;
use dynamis_abi::COUNTER_LIVE;
use dynamis_abi::Count;
use dynamis_gpu::Resources;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::Stage;
use dynamis_shader::{CORE, rows, stream};
use dynamis_sort::RadixSort;
use dynamis_state::StateStream;

pub struct Integrate {
    begin_step: Stage,
    substep_integrate: Stage,
    substep_advance: Stage,
    broadphase_aabb: Stage,
}

impl Integrate {
    pub fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            begin_step: Stage::build(
                context,
                "begin_step",
                rows(
                    context,
                    include_str!("../shaders/begin_step.wgsl"),
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
            substep_integrate: Stage::build(
                context,
                "substep_integrate",
                stream(
                    context,
                    include_str!("../shaders/substep_integrate.wgsl"),
                    CORE,
                    "work",
                    RigidStream::LiveBodies,
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("live_bodies", RigidStream::LiveBodies.whole()),
                    ("live_count", dynamis_state::counter(COUNTER_LIVE)),
                ],
                &[],
            ),
            substep_advance: Stage::build(
                context,
                "substep_advance",
                stream(
                    context,
                    include_str!("../shaders/substep_advance.wgsl"),
                    CORE,
                    "work",
                    RigidStream::LiveBodies,
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("live_bodies", RigidStream::LiveBodies.whole()),
                    ("live_count", dynamis_state::counter(COUNTER_LIVE)),
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

    fn record_begin(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.begin_step
            .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
    }

    pub fn record_substep(&self, recorder: &mut ComputeRecorder, streams: &impl Resources) {
        self.substep_integrate.record_stream(recorder, streams);
    }

    pub fn record_substep_advance(&self, recorder: &mut ComputeRecorder, streams: &impl Resources) {
        self.substep_advance.record_stream(recorder, streams);
    }

    pub fn record_broadphase(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.broadphase_aabb
            .record_rows(recorder, streams, Count::Colliders.rows(&frame.params));
    }

    pub fn record(
        &self,
        recorder: &mut ComputeRecorder,
        streams: &impl Resources,
        frame: &RigidFrame,
        sort: &RadixSort,
    ) {
        if frame.params.constraint_count > 0 {
            let words = frame.shape.body_row_words;
            let channels = sort::lanes_dual(
                streams,
                dynamis_state::counter(COUNTER_JOINTS),
                RigidStream::JointFilterMajor.whole(),
                RigidStream::JointFilterMinor.whole(),
            );
            sort.sort(recorder, &channels, words, words);
        }
        self.record_begin(recorder, streams, frame);
        self.record_broadphase(recorder, streams, frame);
    }
}
