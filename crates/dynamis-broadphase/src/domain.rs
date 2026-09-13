use crate::{Broadphase, BroadphaseDemand, BroadphaseFrame, BroadphaseInputs, BroadphasePasses};
use crate::{BroadphaseCapacity, BroadphaseStreams, Capacity};
use dynamis_abi::Counters;
use dynamis_domain::{Domain, Run, StepFacts};
use dynamis_gpu::GpuContext;
use dynamis_pass::{PassOrder, Phase, Resources, Schedule};
use wgpu::CommandEncoder;

pub struct BroadphaseDomain;

impl Domain for BroadphaseDomain {
    const ID: u32 = 1;

    type Demand = BroadphaseDemand;
    type Inputs = BroadphaseInputs;
    type Work = ();
    type Streams = BroadphaseStreams;
    type Planner = Capacity;
    type Passes = BroadphasePasses;
    type Runtime = Broadphase;
    type Frame = BroadphaseFrame;
    type Capacity = BroadphaseCapacity;

    fn minimum() -> BroadphaseDemand {
        Capacity::floor()
    }

    fn active(_: &Counters, _: &()) -> bool {
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

    fn frame(_: &StepFacts, _: &BroadphaseInputs, run: Run) -> BroadphaseFrame {
        BroadphaseFrame {
            indexing: run.indexing,
        }
    }

    fn capacity(streams: &BroadphaseStreams) -> BroadphaseCapacity {
        crate::capacity(streams)
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
