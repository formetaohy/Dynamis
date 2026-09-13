use crate::capacity::Capacity;
use crate::{
    Ccd, CcdPasses, Rigid, RigidDemand, RigidFrame, RigidInputs, RigidPasses,
    RigidResolutionPasses, RigidShape, RigidStreams,
};
use dynamis_abi::{COUNTER_ACTIVE, Counters};
use dynamis_domain::{Domain, HostWork, Ledger, StepFacts};
use dynamis_gpu::GpuContext;
use dynamis_pass::{PassOrder, Phase, Resources, Schedule};
use wgpu::CommandEncoder;

pub struct RigidDomain;

pub struct RigidDomainPasses {
    pub simulation: RigidPasses,
    pub resolution: RigidResolutionPasses,
    pub continuous: CcdPasses,
}

pub struct RigidDomainRuntime {
    pub simulation: Rigid,
    pub continuous: Ccd,
}

impl Domain for RigidDomain {
    const ID: u32 = 2;

    type Demand = RigidDemand;
    type Inputs = RigidInputs;
    type Streams = RigidStreams;
    type Planner = Capacity;
    type Passes = RigidDomainPasses;
    type Runtime = RigidDomainRuntime;
    type Frame = RigidFrame;

    fn minimum() -> RigidDemand {
        Capacity::floor(dynamis_broadphase::BroadphaseDomain::minimum().pairs)
    }

    fn plan(
        planner: &mut Capacity,
        measured: &Counters,
        inputs: &RigidInputs,
        ledger: &mut Ledger,
        current: &RigidStreams,
    ) -> RigidDemand {
        planner.plan(measured, inputs, ledger.idle(), ledger.pairs(), current)
    }

    fn active(measured: &Counters, work: &HostWork) -> bool {
        measured[COUNTER_ACTIVE] != 0 || work.body_commands > 0 || work.constraint_commands > 0
    }

    fn claim(order: &mut PassOrder) -> RigidDomainPasses {
        RigidDomainPasses {
            simulation: RigidPasses::claim(order),
            resolution: RigidResolutionPasses::claim(order),
            continuous: CcdPasses::claim(order),
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

    fn frame(facts: &StepFacts, inputs: &RigidInputs) -> RigidFrame {
        RigidFrame {
            params: facts.params,
            shape: RigidShape::of(&facts.counts),
            query_count: facts.work.queries,
            simulating: facts.simulating,
            ccd: inputs.ccd,
        }
    }

    fn record(
        runtime: &RigidDomainRuntime,
        phase: Phase,
        schedule: &mut Schedule,
        encoder: &mut CommandEncoder,
        streams: &impl Resources,
        frame: &RigidFrame,
    ) {
        runtime
            .simulation
            .record(phase, schedule, encoder, streams, frame);
        runtime
            .continuous
            .record(phase, schedule, encoder, streams, frame);
    }
}
