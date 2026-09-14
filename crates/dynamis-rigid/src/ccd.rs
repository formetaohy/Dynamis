use crate::RigidFrame;
use crate::RigidStream;
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::Resources;
use dynamis_pass::{Execution, Schedule, Stage, domain_passes};

use dynamis_abi::{COUNTER_PAIRS, Count};
use dynamis_gpu::GpuContext;
use dynamis_shader::{CORE, GEOMETRY, rows, stream};
use dynamis_state::StateStream;

pub const CCD_GATE: u32 = 0;

pub const CCD_EXECUTION: Execution = Execution::AWAKE.and(Execution::gate(CCD_GATE));

domain_passes!(
    CcdPasses,
    ccd_sweep => CCD_EXECUTION => &["substeps"],
    ccd_apply => CCD_EXECUTION => &["ccd_sweep"],
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
        &mut self,
        pass: u32,
        schedule: &mut Schedule,
        encoder: &mut wgpu::CommandEncoder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        if pass == self.passes.ccd_sweep {
            let mut sweep = schedule.open(encoder, pass);
            self.sweep.record_stream(&mut sweep, streams);
            drop(sweep);
        } else if pass == self.passes.ccd_apply {
            let mut apply = schedule.open(encoder, pass);
            self.apply
                .record_rows(&mut apply, streams, Count::Dynamic.rows(&frame.params));
            drop(apply);
        }
    }
}
