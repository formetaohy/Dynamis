use crate::capacity::floor;
use crate::{Soft, SoftCapacity, SoftDemand, SoftFrame, SoftInputs, SoftPasses, SoftStreams};
use dynamis_abi::COUNTER_SOFT_ACTIVE;
use dynamis_abi::Counters;
use dynamis_domain::{Domain, Run, StepFacts};
use dynamis_gpu::GpuContext;
use dynamis_pass::{PassOrder, Phase, Resources, Schedule};
use wgpu::CommandEncoder;

pub struct SoftDomain;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SoftWork {
    pub uploads: bool,
}

impl Domain for SoftDomain {
    const ID: u32 = 3;

    type Demand = SoftDemand;
    type Inputs = SoftInputs;
    type Work = SoftWork;
    type Streams = SoftStreams;
    type Planner = ();
    type Passes = SoftPasses;
    type Runtime = Soft;
    type Frame = SoftFrame;
    type Capacity = SoftCapacity;

    fn minimum() -> SoftDemand {
        floor()
    }

    fn active(measured: &Counters, work: &SoftWork) -> bool {
        measured[COUNTER_SOFT_ACTIVE] != 0 || work.uploads
    }

    fn claim(order: &mut PassOrder) -> SoftPasses {
        SoftPasses::claim(order)
    }

    fn build(context: &GpuContext, streams: &impl Resources, passes: SoftPasses) -> Soft {
        Soft::new(context, streams, passes)
    }

    fn frame(facts: &StepFacts, inputs: &SoftInputs, run: Run) -> SoftFrame {
        SoftFrame {
            params: facts.params,
            simulating: run.awake && inputs.particles > 0,
            material: inputs.material,
        }
    }

    fn capacity(streams: &SoftStreams) -> SoftCapacity {
        crate::capacity(streams)
    }

    fn record(
        runtime: &Soft,
        phase: Phase,
        schedule: &mut Schedule,
        encoder: &mut CommandEncoder,
        streams: &impl Resources,
        frame: &SoftFrame,
    ) {
        runtime.record(phase, schedule, encoder, streams, frame);
    }
}
