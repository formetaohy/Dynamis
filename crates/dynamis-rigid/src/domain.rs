use crate::capacity::Capacity;
use crate::{
    Ccd, CcdPasses, Rigid, RigidCapacity, RigidDemand, RigidFrame, RigidInputs, RigidPasses,
    RigidResolutionPasses, RigidShape, RigidStreams,
};
use dynamis_abi::COUNTER_ACTIVE;
use dynamis_abi::Counters;
use dynamis_domain::{Domain, StepFacts};
use dynamis_gpu::GpuContext;
use dynamis_gpu::Resources;
use dynamis_pass::{Execution, PassGroup, Pipeline, Schedule};
use wgpu::CommandEncoder;

pub struct RigidDomain;

pub struct RigidDomainPasses {
    pub simulation: RigidPasses,
    pub continuous: CcdPasses,
    pub resolution: RigidResolutionPasses,
}

pub struct RigidDomainRuntime {
    pub simulation: Rigid,
    pub continuous: Ccd,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RigidWork {
    pub body_commands: u32,
    pub constraint_commands: u32,
}

impl Domain for RigidDomain {
    const ID: u32 = 2;

    const SIMULATES: bool = true;

    const PASS_EDGES: dynamis_pass::PassEdges = &[
        RigidPasses::EDGES,
        CcdPasses::EDGES,
        RigidResolutionPasses::EDGES,
    ];

    type Demand = RigidDemand;
    type Inputs = RigidInputs;
    type Work = RigidWork;
    type Streams = RigidStreams;
    type Planner = Capacity;
    type Passes = RigidDomainPasses;
    type Runtime = RigidDomainRuntime;
    type Frame = RigidFrame;
    type Capacity = RigidCapacity;

    fn minimum() -> RigidDemand {
        Capacity::floor(dynamis_domain::STREAM_FLOOR)
    }

    fn occupied(inputs: &RigidInputs) -> bool {
        inputs.bodies > 0
    }

    fn pending(work: &RigidWork) -> bool {
        work.body_commands > 0 || work.constraint_commands > 0
    }

    fn active(measured: &Counters) -> bool {
        measured[COUNTER_ACTIVE] != 0
    }

    fn pass_groups() -> &'static [PassGroup] {
        &[
            RigidPasses::GROUP,
            CcdPasses::GROUP,
            RigidResolutionPasses::GROUP,
        ]
    }

    fn resolve(pipeline: &Pipeline) -> RigidDomainPasses {
        RigidDomainPasses {
            simulation: RigidPasses::resolve(pipeline),
            continuous: CcdPasses::resolve(pipeline),
            resolution: RigidResolutionPasses::resolve(pipeline),
        }
    }

    fn build(
        context: &GpuContext,
        streams: &impl Resources,
        passes: RigidDomainPasses,
    ) -> RigidDomainRuntime {
        RigidDomainRuntime {
            simulation: Rigid::new(context, streams, passes.simulation, passes.resolution),
            continuous: Ccd::new(context, streams, passes.continuous),
        }
    }

    fn gates(frame: &RigidFrame) -> u16 {
        if frame.ccd {
            Execution::gate(crate::ccd::CCD_GATE).bits()
        } else {
            0
        }
    }

    fn frame(facts: &StepFacts, inputs: &RigidInputs) -> RigidFrame {
        RigidFrame {
            params: facts.params,
            shape: RigidShape::of(&facts.counts),
            query_count: inputs.queries,
            observed_count: inputs.observed,
            ccd: inputs.ccd,
        }
    }

    fn capacity(streams: &RigidStreams) -> RigidCapacity {
        crate::capacity(streams)
    }

    fn record(
        runtime: &mut RigidDomainRuntime,
        pass: u32,
        schedule: &mut Schedule,
        encoder: &mut CommandEncoder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        runtime
            .simulation
            .record(pass, schedule, encoder, streams, frame);
        runtime
            .continuous
            .record(pass, schedule, encoder, streams, frame);
    }
}
