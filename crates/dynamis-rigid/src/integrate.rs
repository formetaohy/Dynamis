use super::streams::RigidStream;
use crate::RigidFrame;
use crate::sort;
use dynamis_abi::COUNTER_JOINTS;
use dynamis_abi::COUNTER_LIVE;
use dynamis_abi::Count;
use dynamis_gpu::Resources;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use dynamis_pass::{PassRuntime, Stage};
use dynamis_shader::{CORE, rows, stream};
use dynamis_sort::RadixSort;
use dynamis_state::StateStream;

fn aabb(context: &GpuContext, streams: &impl Resources) -> Stage {
    Stage::build(
        context,
        "broadphase_aabb",
        rows(
            context,
            include_str!("../shaders/broadphase_aabb.wgsl"),
            CORE,
            Count::Colliders.bound(),
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
    )
}

pub struct Prepare {
    begin_step: Stage,
    aabb: Stage,
    sort: RadixSort,
}

impl Prepare {
    fn sort_joints(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        if frame.params.constraint_count > 0 {
            let words = frame.shape.body_row_words;
            let channels = sort::lanes_dual(
                streams,
                dynamis_state::counter(COUNTER_JOINTS),
                RigidStream::JointFilterMajor.whole(),
                RigidStream::JointFilterMinor.whole(),
            );
            self.sort.sort(recorder, &channels, words, words);
        }
    }
}

impl PassRuntime<RigidFrame> for Prepare {
    fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            begin_step: Stage::build(
                context,
                "begin_step",
                rows(
                    context,
                    include_str!("../shaders/begin_step.wgsl"),
                    CORE,
                    Count::Dynamic.bound(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("ccd_factor", RigidStream::CcdFactor.whole()),
                ],
                &[],
            ),
            aabb: aabb(context, streams),
            sort: RadixSort::new(context),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.sort_joints(recorder, streams, frame);
        self.begin_step.record_rows(
            recorder,
            streams,
            Count::Dynamic.rows(&frame.params, &frame.rows),
        );
        self.aabb.record_rows(
            recorder,
            streams,
            Count::Colliders.rows(&frame.params, &frame.rows),
        );
    }
}

pub struct QueryAabbs {
    aabb: Stage,
}

impl PassRuntime<RigidFrame> for QueryAabbs {
    fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            aabb: aabb(context, streams),
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.aabb.record_rows(
            recorder,
            streams,
            Count::Colliders.rows(&frame.params, &frame.rows),
        );
    }
}

pub(crate) struct SubstepIntegrate {
    integrate: Stage,
    advance: Stage,
}

impl SubstepIntegrate {
    pub(crate) fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            integrate: Stage::build(
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
            advance: Stage::build(
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
        }
    }

    pub(crate) fn record(&mut self, recorder: &mut ComputeRecorder<'_>, streams: &impl Resources) {
        self.integrate.record_stream(recorder, streams);
    }

    pub(crate) fn record_advance(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
    ) {
        self.advance.record_stream(recorder, streams);
    }
}
