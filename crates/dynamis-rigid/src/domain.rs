use crate::{
    CcdPasses, CcdRuntime, RigidCapacity, RigidDemand, RigidFrame, RigidInputs, RigidPasses,
    RigidResolutionPasses, RigidResolutionRuntime, RigidRuntime, RigidShape, RigidStreams,
};
use dynamis_abi::COUNTER_ACTIVE;
use dynamis_abi::Counters;
use dynamis_domain::{Domain, StepFacts};
use dynamis_gpu::ComputeRecorder;
use dynamis_gpu::GpuContext;
use dynamis_gpu::ResourceSource;
use dynamis_pass::{Execution, PassGroup, Pipeline, domain_groups};

pub struct RigidDomain;

domain_groups!(
    RigidDomainPasses,
    RigidDomainRuntime,
    RigidFrame,
    simulation: RigidPasses => RigidRuntime,
    continuous: CcdPasses => CcdRuntime,
    resolution: RigidResolutionPasses => RigidResolutionRuntime,
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RigidWork {
    pub body_commands: u32,
    pub constraint_commands: u32,
    pub character_inputs: bool,
    pub vehicle_inputs: bool,
    pub wake_all: bool,
}

impl Domain for RigidDomain {
    const ID: u32 = 2;

    const SIMULATES: bool = true;

    const PASS_EDGES: dynamis_pass::PassEdges = RigidDomainPasses::EDGES;

    type Demand = RigidDemand;
    type Inputs = RigidInputs;
    type Work = RigidWork;
    type Streams = RigidStreams;
    type Passes = RigidDomainPasses;
    type Runtime = RigidDomainRuntime;
    type Frame = RigidFrame;
    type Capacity = RigidCapacity;

    fn minimum() -> RigidDemand {
        crate::capacity::floor(dynamis_domain::STREAM_FLOOR)
    }

    fn occupied(inputs: &RigidInputs) -> bool {
        inputs.bodies > 0
    }

    fn pending(work: &RigidWork) -> bool {
        work.body_commands > 0
            || work.constraint_commands > 0
            || work.character_inputs
            || work.vehicle_inputs
            || work.wake_all
    }

    fn active(measured: &Counters) -> bool {
        measured[COUNTER_ACTIVE] != 0
    }

    fn pass_groups() -> &'static [PassGroup] {
        RigidDomainPasses::GROUPS
    }

    fn resolve(pipeline: &Pipeline) -> RigidDomainPasses {
        RigidDomainPasses::resolve(pipeline)
    }

    fn build(
        context: &GpuContext,
        streams: &impl ResourceSource,
        passes: RigidDomainPasses,
    ) -> RigidDomainRuntime {
        RigidDomainRuntime::build(context, streams, passes)
    }

    fn gates(frame: &RigidFrame) -> u16 {
        let mut gates = 0;
        if frame.ccd {
            gates |= Execution::gate(crate::ccd::CCD_GATE).bits();
        }
        if frame.observed_joints > 0 {
            gates |= Execution::gate(crate::commit::OBSERVED_JOINTS_GATE).bits();
        }
        if frame.params.wake_all != 0 {
            gates |= Execution::gate(crate::wake::WAKE_ALL_GATE).bits();
        }
        gates
    }

    fn frame(facts: &StepFacts, inputs: &RigidInputs) -> RigidFrame {
        RigidFrame {
            params: facts.params,
            rows: facts.rows,
            shape: RigidShape::of(&facts.census),
            observed_count: inputs.observed,
            observed_joints: inputs.observed_joints,
            joint_groups: inputs.joint_groups,
            ccd: inputs.ccd,
            impacts: inputs.impacts,
            immovable_rebuild: inputs.immovable_rebuild,
        }
    }

    fn capacity(streams: &RigidStreams) -> RigidCapacity {
        crate::capacity(streams)
    }

    fn record(
        runtime: &mut RigidDomainRuntime,
        pass: u32,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &RigidFrame,
    ) -> bool {
        runtime.record(pass, recorder, streams, frame)
    }
}
