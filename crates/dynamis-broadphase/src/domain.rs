use crate::{BroadphaseCapacity, BroadphaseDemand, BroadphaseInputs};
use crate::{BroadphasePasses, BroadphaseRuntime, BroadphaseStreams};
use dynamis_abi::Counters;
use dynamis_domain::{Domain, StepFacts};
use dynamis_gpu::ComputeRecorder;
use dynamis_gpu::GpuContext;
use dynamis_gpu::ResourceSource;
use dynamis_pass::{PassGroup, Pipeline};

pub struct BroadphaseDomain;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BroadphaseFrame {
    pub entry_base: u32,
    pub moving_slots: u32,
    pub immovable_rebuild: bool,
    pub resting_rebuild: bool,
}

impl Domain for BroadphaseDomain {
    const ID: u32 = 1;

    const SIMULATES: bool = false;

    const PASS_EDGES: dynamis_pass::PassEdges = &[BroadphasePasses::EDGES];

    type Demand = BroadphaseDemand;
    type Inputs = BroadphaseInputs;
    type Work = ();
    type Streams = BroadphaseStreams;
    type Passes = BroadphasePasses;
    type Runtime = BroadphaseRuntime;
    type Frame = BroadphaseFrame;
    type Capacity = BroadphaseCapacity;

    fn minimum() -> BroadphaseDemand {
        crate::floor()
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
        streams: &impl ResourceSource,
        passes: BroadphasePasses,
    ) -> BroadphaseRuntime {
        BroadphaseRuntime::build(context, streams, passes)
    }

    fn gates(_: &BroadphaseFrame) -> u16 {
        0
    }

    fn frame(_: &StepFacts, inputs: &BroadphaseInputs) -> BroadphaseFrame {
        BroadphaseFrame {
            entry_base: inputs.entry_base,
            moving_slots: inputs.moving_slots,
            immovable_rebuild: inputs.immovable_rebuild,
            resting_rebuild: inputs.resting_rebuild,
        }
    }

    fn capacity(streams: &BroadphaseStreams) -> BroadphaseCapacity {
        crate::capacity(streams)
    }

    fn record(
        runtime: &mut BroadphaseRuntime,
        pass: u32,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &BroadphaseFrame,
    ) -> bool {
        runtime.record(pass, recorder, streams, frame)
    }
}
