use crate::{Broadphase, BroadphaseDemand, BroadphaseFrame, BroadphaseInputs, BroadphasePasses};
use crate::{BroadphaseStreams, Capacity};
use dynamis_abi::Counters;
use dynamis_domain::{Domain, HostWork, Ledger, StepFacts};
use dynamis_gpu::GpuContext;
use dynamis_pass::{PassOrder, Phase, Resources, Schedule};
use wgpu::CommandEncoder;

pub struct BroadphaseDomain;

impl Domain for BroadphaseDomain {
    const ID: u32 = 1;

    type Demand = BroadphaseDemand;
    type Inputs = BroadphaseInputs;
    type Streams = BroadphaseStreams;
    type Planner = Capacity;
    type Passes = BroadphasePasses;
    type Runtime = Broadphase;
    type Frame = BroadphaseFrame;

    fn minimum() -> BroadphaseDemand {
        Capacity::floor()
    }

    fn plan(
        planner: &mut Capacity,
        measured: &Counters,
        inputs: &BroadphaseInputs,
        ledger: &mut Ledger,
        current: &BroadphaseStreams,
    ) -> BroadphaseDemand {
        let (demand, idle) = planner.plan(measured, inputs, current);
        ledger.set_idle(idle);
        ledger.set_pairs(demand.pairs);
        demand
    }

    fn active(_: &Counters, _: &HostWork) -> bool {
        false
    }

    fn claim(order: &mut PassOrder) -> BroadphasePasses {
        BroadphasePasses::claim(order)
    }

    fn build(
        context: &GpuContext,
        streams: &impl Resources,
        passes: BroadphasePasses,
    ) -> Broadphase {
        Broadphase::new(context, streams, passes)
    }

    fn frame(facts: &StepFacts, _: &BroadphaseInputs) -> BroadphaseFrame {
        BroadphaseFrame {
            indexing: facts.indexing,
        }
    }

    fn record(
        runtime: &Broadphase,
        phase: Phase,
        schedule: &mut Schedule,
        encoder: &mut CommandEncoder,
        streams: &impl Resources,
        frame: &BroadphaseFrame,
    ) {
        runtime.record(phase, schedule, encoder, streams, *frame);
    }
}
