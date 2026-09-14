use crate::{Broadphase, BroadphaseDemand, BroadphaseFrame, BroadphaseInputs, BroadphasePasses};
use crate::{BroadphaseCapacity, BroadphaseStreams, Capacity};
use dynamis_abi::Counters;
use dynamis_domain::{Domain, Run, StepFacts};
use dynamis_gpu::GpuContext;
use dynamis_gpu::Resources;
use dynamis_pass::{PassGroup, Pipeline, Schedule};
use wgpu::CommandEncoder;

pub struct BroadphaseDomain;

impl Domain for BroadphaseDomain {
    const ID: u32 = 1;

    const SIMULATES: bool = false;

    const PASS_EDGES: dynamis_pass::PassEdges = &[BroadphasePasses::EDGES];

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

    fn occupied(inputs: &BroadphaseInputs) -> bool {
        inputs.colliders > 0 || inputs.particles > 0
    }

    fn pending(_: &()) -> bool {
        false
    }

    fn active(_: &Counters) -> bool {
        false
    }

    fn pass_groups() -> &'static [PassGroup] {
        &[BroadphasePasses::GROUP]
    }

    fn resolve(pipeline: &Pipeline) -> BroadphasePasses {
        BroadphasePasses::resolve(pipeline)
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
        pass: u32,
        schedule: &mut Schedule,
        encoder: &mut CommandEncoder,
        streams: &impl Resources,
        frame: &BroadphaseFrame,
    ) {
        runtime.record(pass, schedule, encoder, streams, *frame);
    }
}
