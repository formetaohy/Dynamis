use crate::RigidFrame;
use crate::RigidStream;
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::{ComputeRecorder, Resources};
use dynamis_pass::{Execution, Stage, domain_passes};

use dynamis_abi::{COUNTER_JOINTS, COUNTER_PAIRS, Count};
use dynamis_gpu::GpuContext;
use dynamis_shader::{CORE, rows, stream};
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
                    &crate::geometry_fragments(),
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
                    ("joint_major", RigidStream::JointFilterMajor.whole()),
                    ("joint_minor", RigidStream::JointFilterMinor.whole()),
                    ("joint_count", dynamis_state::counter(COUNTER_JOINTS)),
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
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) -> bool {
        if pass == self.passes.ccd_sweep {
            self.sweep.record_stream(recorder, streams);
        } else if pass == self.passes.ccd_apply {
            self.apply
                .record_rows(recorder, streams, Count::Dynamic.rows(&frame.params));
        } else {
            return false;
        }
        true
    }
}
