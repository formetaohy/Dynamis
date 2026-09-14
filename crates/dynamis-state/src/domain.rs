use crate::capacity::floor;
use crate::{StateCapacity, StateDemand, StateInputs, StateStreams};
use dynamis_abi::Counters;
use dynamis_domain::{Domain, StepFacts};
use dynamis_gpu::GpuContext;
use dynamis_gpu::Resources;
use dynamis_pass::{PassGroup, Pipeline, Schedule};
use wgpu::CommandEncoder;

pub struct StateDomain;

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
    type Passes = ();
    type Runtime = ();
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
        &[]
    }

    fn resolve(_: &Pipeline) {}

    fn build(_: &GpuContext, _: &impl Resources, _: ()) {}

    fn gates(_: &()) -> u16 {
        0
    }

    fn frame(_: &StepFacts, _: &StateInputs) {}

    fn capacity(streams: &StateStreams) -> StateCapacity {
        crate::capacity(streams)
    }

    fn record(
        _: &mut (),
        _: u32,
        _: &mut Schedule,
        _: &mut CommandEncoder,
        _: &impl Resources,
        _: &(),
    ) {
    }
}
