use crate::capacity::floor;
use crate::{StateCapacity, StateDemand, StateInputs, StateStream, StateStreams};
use dynamis_abi::Counters;
use dynamis_domain::{Domain, StepFacts};
use dynamis_gpu::ComputeRecorder;
use dynamis_gpu::GpuContext;
use dynamis_gpu::Resources;
use dynamis_pass::{Execution, PassGroup, Pipeline, Stage, domain_passes};

pub struct StateDomain;

domain_passes!(
    StatePasses,
    consume_streams => Execution::STEP => &["commit"],
);

pub struct StateRuntime {
    consume_streams: Stage,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StateWork {
    pub queries: u32,
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
    type Planner = ();
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
        work.queries > 0 || work.shape_uploads
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

    fn build(context: &GpuContext, streams: &impl Resources, _: StatePasses) -> StateRuntime {
        StateRuntime {
            consume_streams: Stage::build(
                context,
                "consume_streams",
                dynamis_shader::workgroups(
                    context,
                    include_str!("../shaders/consume_streams.wgsl"),
                    dynamis_shader::CORE,
                ),
                streams,
                &[
                    ("row_streams", StateStream::RowStreams.whole()),
                    ("counters", StateStream::Counters.whole()),
                ],
                &[],
            ),
        }
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
        _: u32,
        recorder: &mut ComputeRecorder<'_>,
        streams: &impl Resources,
        _: &(),
    ) -> bool {
        runtime
            .consume_streams
            .record_workgroups(recorder, streams, 1);
        true
    }
}
