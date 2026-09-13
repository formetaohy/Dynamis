use crate::capacity::floor;
use crate::{StateCapacity, StateDemand, StateInputs, StateStreams};
use dynamis_abi::Counters;
use dynamis_domain::{Domain, Run, StepFacts};
use dynamis_gpu::GpuContext;
use dynamis_pass::{PassGroup, Pipeline, Resources, Schedule};
use wgpu::CommandEncoder;

pub struct StateDomain;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StateWork {
    pub queries: u32,
    pub shape_uploads: bool,
}

impl Domain for StateDomain {
    const ID: u32 = 0;

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

    fn active(_: &Counters, work: &StateWork) -> bool {
        work.queries > 0 || work.shape_uploads
    }

    fn pass_groups() -> &'static [PassGroup] {
        &[]
    }

    fn resolve(_: &Pipeline) {}

    fn build(_: &GpuContext, _: &impl Resources, _: ()) {}

    fn frame(_: &StepFacts, _: &StateInputs, _: Run) {}

    fn capacity(streams: &StateStreams) -> StateCapacity {
        crate::capacity(streams)
    }

    fn record(
        _: &(),
        _: u32,
        _: &mut Schedule,
        _: &mut CommandEncoder,
        _: &impl Resources,
        _: &(),
    ) {
    }
}
