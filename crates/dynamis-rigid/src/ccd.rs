use crate::RigidStream;
use dynamis_broadphase::BroadphaseStream;
use dynamis_pass::Resources;
use dynamis_pass::{Phase, Schedule, Stage, domain_passes};
use dynamis_state::{Count, StateStream, StepFrame};

use dynamis_abi::COUNTER_PAIRS;
use dynamis_gpu::GpuContext;
use dynamis_kernel::{CORE, GEOMETRY, rows, stream};

domain_passes!(
    CcdPasses,
    "ccd",
    sweep: Phase::Continuous => "ccd_sweep",
    apply: Phase::Continuous => "ccd_apply",
);

pub struct Ccd {
    passes: CcdPasses,
    sweep: Stage,
    apply: Stage,
}

impl Ccd {
    pub fn new(context: &GpuContext, streams: &impl Resources, passes: CcdPasses) -> Self {
        Self {
            passes,
            sweep: Stage::build(
                context,
                "ccd_sweep",
                stream(
                    context,
                    include_str!("../shaders/ccd_sweep.wgsl"),
                    GEOMETRY,
                    "work",
                    BroadphaseStream::PairMajor,
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("body_descs", StateStream::BodyDescriptors.whole()),
                    ("colliders", StateStream::Colliders.whole()),
                    ("pair_major", BroadphaseStream::PairMajor.whole()),
                    ("pair_minor", BroadphaseStream::PairMinor.whole()),
                    ("pair_count", dynamis_state::counter(COUNTER_PAIRS)),
                    ("collider_owners", StateStream::ColliderOwners.whole()),
                    ("ccd_factor", RigidStream::CcdFactor.whole()),
                    ("ccd_impact", RigidStream::CcdImpact.whole()),
                ],
                &dynamis_state::shape_resources(),
            ),
            apply: Stage::build(
                context,
                "ccd_apply",
                rows(
                    context,
                    include_str!("../shaders/ccd_apply.wgsl"),
                    CORE,
                    Count::Dynamic.field(),
                ),
                streams,
                &[
                    ("params", StateStream::Params.whole()),
                    ("body_states", StateStream::BodyStates.whole()),
                    ("ccd_factor", RigidStream::CcdFactor.whole()),
                    ("ccd_impact", RigidStream::CcdImpact.whole()),
                ],
                &[],
            ),
        }
    }

    pub fn record(
        &self,
        phase: Phase,
        schedule: &mut Schedule,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
        frame: &StepFrame,
    ) {
        if phase != Phase::Continuous || !frame.ccd_bodies || !frame.simulating() {
            return;
        }
        let mut sweep = schedule.open(encoder, self.passes.sweep);
        self.sweep.record_stream(&mut sweep, streams);
        drop(sweep);

        let mut apply = schedule.open(encoder, self.passes.apply);
        self.apply
            .record_rows(&mut apply, streams, Count::Dynamic.rows(&frame.params));
        drop(apply);
    }
}
