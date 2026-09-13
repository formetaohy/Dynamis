use crate::capacity::{floor, plan};
use crate::{StateDemand, StateInputs, StateStreams};
use dynamis_abi::Counters;
use dynamis_domain::{Domain, HostWork, Ledger, StepFacts};
use dynamis_gpu::GpuContext;
use dynamis_pass::{PassOrder, Phase, Resources, Schedule};
use wgpu::CommandEncoder;

pub struct StateDomain;

impl Domain for StateDomain {
    const ID: u32 = 0;

    type Demand = StateDemand;
    type Inputs = StateInputs;
    type Streams = StateStreams;
    type Planner = ();
    type Passes = ();
    type Runtime = ();
    type Frame = ();

    fn minimum() -> StateDemand {
        floor()
    }

    fn plan(
        _: &mut (),
        _: &Counters,
        inputs: &StateInputs,
        ledger: &mut Ledger,
        current: &StateStreams,
    ) -> StateDemand {
        let demand = plan(inputs, ledger.idle(), current);
        ledger.set_bodies(demand.bodies);
        demand
    }

    fn active(_: &Counters, work: &HostWork) -> bool {
        work.queries > 0 || work.shape_uploads
    }

    fn claim(_: &mut PassOrder) {}

    fn build(_: &GpuContext, _: &impl Resources, _: ()) {}

    fn frame(_: &StepFacts, _: &StateInputs) {}

    fn record(
        _: &(),
        _: Phase,
        _: &mut Schedule,
        _: &mut CommandEncoder,
        _: &impl Resources,
        _: &(),
    ) {
    }
}
