use crate::capacity::floor;
use crate::{Soft, SoftCapacity, SoftDemand, SoftFrame, SoftInputs, SoftPasses, SoftStreams};
use dynamis_abi::COUNTER_SOFT_ACTIVE;
use dynamis_abi::Counters;
use dynamis_domain::{Domain, StepFacts};
use dynamis_gpu::ComputeRecorder;
use dynamis_gpu::GpuContext;
use dynamis_gpu::Resources;
use dynamis_pass::{PassGroup, Pipeline};

pub struct SoftDomain;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SoftWork {
    pub uploads: bool,
    pub edits: u32,
    pub forces: bool,
}

impl Domain for SoftDomain {
    const ID: u32 = 3;

    const SIMULATES: bool = true;

    const PASS_EDGES: dynamis_pass::PassEdges = &[SoftPasses::EDGES];

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

    fn occupied(inputs: &SoftInputs) -> bool {
        inputs.particles > 0
    }

    fn pending(work: &SoftWork) -> bool {
        work.uploads || work.edits > 0 || work.forces
    }

    fn active(measured: &Counters) -> bool {
        measured[COUNTER_SOFT_ACTIVE] != 0
    }

    fn pass_groups() -> &'static [PassGroup] {
        &[SoftPasses::GROUP]
    }

    fn resolve(pipeline: &Pipeline) -> SoftPasses {
        SoftPasses::resolve(pipeline)
    }

    fn build(context: &GpuContext, streams: &impl Resources, passes: SoftPasses) -> Soft {
        Soft::new(context, streams, passes)
    }

    fn gates(_: &SoftFrame) -> u16 {
        0
    }

    fn frame(facts: &StepFacts, inputs: &SoftInputs) -> SoftFrame {
        SoftFrame {
            params: facts.params,
            material: inputs.material,
        }
    }

    fn capacity(streams: &SoftStreams) -> SoftCapacity {
        crate::capacity(streams)
    }

    fn record(
        runtime: &mut Soft,
        pass: u32,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        frame: &SoftFrame,
    ) -> bool {
        runtime.record(pass, recorder, streams, frame)
    }
}
