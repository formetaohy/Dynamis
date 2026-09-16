use crate::capacity::floor;
use crate::passes::WAKE_ALL_GATE;
use crate::{
    SoftCapacity, SoftDemand, SoftFrame, SoftInputs, SoftPasses, SoftRuntime, SoftStreams,
};
use dynamis_abi::COUNTER_SOFT_ACTIVE;
use dynamis_abi::Counters;
use dynamis_domain::{Domain, StepFacts};
use dynamis_gpu::ComputeRecorder;
use dynamis_gpu::GpuContext;
use dynamis_gpu::ResourceSource;
use dynamis_pass::{Execution, PassGroup, Pipeline};

pub struct SoftDomain;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SoftWork {
    pub uploads: bool,
    pub body_edits: u32,
    pub edits: u32,
    pub wake_all: bool,
}

impl Domain for SoftDomain {
    const ID: u32 = 3;

    const SIMULATES: bool = true;

    const PASS_EDGES: dynamis_pass::PassEdges = &[SoftPasses::EDGES];

    type Demand = SoftDemand;
    type Inputs = SoftInputs;
    type Work = SoftWork;
    type Streams = SoftStreams;
    type Passes = SoftPasses;
    type Runtime = SoftRuntime;
    type Frame = SoftFrame;
    type Capacity = SoftCapacity;

    fn minimum() -> SoftDemand {
        floor()
    }

    fn occupied(inputs: &SoftInputs) -> bool {
        inputs.particles > 0
    }

    fn pending(work: &SoftWork) -> bool {
        work.uploads || work.body_edits > 0 || work.edits > 0 || work.wake_all
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

    fn build(
        context: &GpuContext,
        streams: &impl ResourceSource,
        passes: SoftPasses,
    ) -> SoftRuntime {
        SoftRuntime::build(context, streams, passes)
    }

    fn gates(frame: &SoftFrame) -> u16 {
        if frame.params.wake_all != 0 {
            return Execution::gate(WAKE_ALL_GATE).bits();
        }
        0
    }

    fn frame(facts: &StepFacts, inputs: &SoftInputs) -> SoftFrame {
        SoftFrame {
            params: facts.params,
            rows: facts.rows,
            material: inputs.material,
        }
    }

    fn capacity(streams: &SoftStreams) -> SoftCapacity {
        crate::capacity(streams)
    }

    fn record(
        runtime: &mut SoftRuntime,
        pass: u32,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &SoftFrame,
    ) -> bool {
        runtime.record(pass, recorder, streams, frame)
    }
}
