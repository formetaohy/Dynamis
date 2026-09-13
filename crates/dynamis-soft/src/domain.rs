use crate::capacity::{floor, plan};
use crate::{Soft, SoftDemand, SoftFrame, SoftInputs, SoftPasses, SoftStreams};
use dynamis_abi::Counters;
use dynamis_domain::{Domain, HostWork, Ledger, StepFacts};
use dynamis_gpu::GpuContext;
use dynamis_pass::{PassOrder, Phase, Resources, Schedule};
use wgpu::CommandEncoder;

pub struct SoftDomain;

impl Domain for SoftDomain {
    const ID: u32 = 3;

    type Demand = SoftDemand;
    type Inputs = SoftInputs;
    type Streams = SoftStreams;
    type Planner = ();
    type Passes = SoftPasses;
    type Runtime = Soft;
    type Frame = SoftFrame;

    fn minimum() -> SoftDemand {
        floor()
    }

    fn plan(
        _: &mut (),
        _: &Counters,
        inputs: &SoftInputs,
        ledger: &mut Ledger,
        current: &SoftStreams,
    ) -> SoftDemand {
        plan(inputs, ledger.idle(), ledger.bodies(), current)
    }

    fn active(_: &Counters, work: &HostWork) -> bool {
        work.soft_bodies > 0
    }

    fn claim(order: &mut PassOrder) -> SoftPasses {
        SoftPasses::claim(order)
    }

    fn build(context: &GpuContext, streams: &impl Resources, passes: SoftPasses) -> Soft {
        Soft::new(context, streams, passes)
    }

    fn frame(facts: &StepFacts, inputs: &SoftInputs) -> SoftFrame {
        SoftFrame {
            params: facts.params,
            simulating: facts.soft && facts.work.soft_bodies > 0,
            material: inputs.material,
        }
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
