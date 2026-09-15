use crate::RigidFrame;
use crate::RigidStream;
use dynamis_broadphase::BroadphaseStream;
use dynamis_gpu::{ComputeRecorder, Resources};
use dynamis_pass::{Execution, PassRuntime, Stage, domain_passes};

use dynamis_abi::{COUNTER_JOINTS, COUNTER_PAIRS, Count};
use dynamis_gpu::GpuContext;
use dynamis_shader::{CORE, rows, stream};
use dynamis_state::StateStream;

pub const CCD_GATE: u32 = 0;

pub const CCD_EXECUTION: Execution = Execution::AWAKE.and(Execution::gate(CCD_GATE));

pub struct CcdSweep {
    sweep: Stage,
}

pub struct CcdApply {
    apply: Stage,
}

impl PassRuntime<RigidFrame> for CcdSweep {
    fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
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
        }
    }

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        _: &RigidFrame,
    ) {
        self.sweep.record_stream(recorder, streams);
    }
}

impl PassRuntime<RigidFrame> for CcdApply {
    fn build(context: &GpuContext, streams: &impl Resources) -> Self {
        Self {
            apply: Stage::build(
                context,
                "ccd_apply",
                rows(
                    context,
                    include_str!("../shaders/ccd_apply.wgsl"),
                    CORE,
                    Count::Dynamic.bound(),
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

    fn record(
        &mut self,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        self.apply.record_rows(
            recorder,
            streams,
            Count::Dynamic.rows(&frame.params, &frame.rows),
        );
    }
}

domain_passes!(
    CcdPasses,
    CcdRuntime,
    RigidFrame,
    ccd_sweep: CcdSweep => CCD_EXECUTION => &["substeps"],
    ccd_apply: CcdApply => CCD_EXECUTION => &["ccd_sweep"],
);
