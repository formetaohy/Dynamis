use crate::capacity::floor;
use crate::{
    SceneCapacity, SceneDemand, SceneFrame, SceneInputs, ScenePasses, SceneRuntime, SceneStreams,
};
use dynamis_abi::Counters;
use dynamis_domain::{Domain, StepFacts};
use dynamis_gpu::{ComputeRecorder, GpuContext, ResourceSource};
use dynamis_pass::{PassGroup, Pipeline};

pub struct SceneDomain;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneWork {
    pub queries: u32,
}

impl Domain for SceneDomain {
    const ID: u32 = 4;

    const SIMULATES: bool = false;

    const PASS_EDGES: dynamis_pass::PassEdges = &[ScenePasses::EDGES];

    type Demand = SceneDemand;
    type Inputs = SceneInputs;
    type Work = SceneWork;
    type Streams = SceneStreams;
    type Passes = ScenePasses;
    type Runtime = SceneRuntime;
    type Frame = SceneFrame;
    type Capacity = SceneCapacity;

    fn minimum() -> SceneDemand {
        floor()
    }

    fn occupied(inputs: &SceneInputs) -> bool {
        inputs.queries > 0
    }

    fn pending(work: &SceneWork) -> bool {
        work.queries > 0
    }

    fn active(_: &Counters) -> bool {
        false
    }

    fn pass_groups() -> &'static [PassGroup] {
        &[ScenePasses::GROUP]
    }

    fn resolve(pipeline: &Pipeline) -> ScenePasses {
        ScenePasses::resolve(pipeline)
    }

    fn build(
        context: &GpuContext,
        streams: &impl ResourceSource,
        passes: ScenePasses,
    ) -> SceneRuntime {
        SceneRuntime::build(context, streams, passes)
    }

    fn gates(_: &SceneFrame) -> u16 {
        0
    }

    fn frame(_: &StepFacts, inputs: &SceneInputs) -> SceneFrame {
        SceneFrame {
            query_count: inputs.queries,
        }
    }

    fn capacity(streams: &SceneStreams) -> SceneCapacity {
        crate::capacity(streams)
    }

    fn record(
        runtime: &mut SceneRuntime,
        pass: u32,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl ResourceSource,
        frame: &SceneFrame,
    ) -> bool {
        runtime.record(pass, recorder, streams, frame)
    }
}
