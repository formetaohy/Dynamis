use crate::capacity::floor;
use crate::passes::{StatePasses, StateRuntime};
use crate::{StateCapacity, StateDemand, StateInputs, StateStreams};
use dynamis_abi::Counters;
use dynamis_domain::{Domain, StepFacts};
use dynamis_gpu::ComputeRecorder;
use dynamis_gpu::GpuContext;
use dynamis_gpu::ResourceSource;
use dynamis_pass::{PassGroup, Pipeline};

pub struct StateDomain;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StateWork {
    pub shape_uploads: bool,
}

impl Domain for StateDomain {
    const ID: u32 = 0;

    const SIMULATES: bool = false;

    const PASS_EDGES: dynamis_pass::PassEdges = &[];

    type Demand = StateDemand;
    type Inputs = StateInputs;
    type Work = StateWork;
    type Streams = StateStreams;
    type Passes = StatePasses;
    type Runtime = StateRuntime;
    type Frame = ();
    type Capacity = StateCapacity;

    fn minimum() -> StateDemand {
        floor()
    }

    fn occupied(_: &StateInputs) -> bool {
        true
    }

    fn pending(work: &StateWork) -> bool {
        work.shape_uploads
    }

    fn active(_: &Counters) -> bool {
        false
    }

    fn pass_groups() -> &'static [PassGroup] {
        &[StatePasses::GROUP]
    }

    fn resolve(pipeline: &Pipeline) -> StatePasses {
        StatePasses::resolve(pipeline)
    }

    fn build(
        context: &GpuContext,
        streams: &impl ResourceSource,
        passes: StatePasses,
    ) -> StateRuntime {
        StateRuntime::build(context, streams, passes)
    }

    fn gates(_: &()) -> u16 {
        0
    }

    fn frame(_: &StepFacts, _: &StateInputs) {}

    fn capacity(streams: &StateStreams) -> StateCapacity {
        crate::capacity(streams)
    }

    fn record(
        runtime: &mut StateRuntime,
        pass: u32,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        _: &(),
    ) -> bool {
        runtime.record(pass, recorder, streams, &())
    }
}
