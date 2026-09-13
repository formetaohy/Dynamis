mod ledger;
mod work;

pub use ledger::Ledger;
pub use work::{HostWork, StepFacts};

use dynamis_abi::Counters;
use dynamis_gpu::GpuContext;
use dynamis_pass::{PassOrder, Phase, Resources, Schedule};
use wgpu::CommandEncoder;

pub trait Domain {
    const ID: u32;

    type Demand: Copy;
    type Inputs: Copy;
    type Streams;
    type Planner;
    type Passes;
    type Runtime;
    type Frame: Copy;

    fn minimum() -> Self::Demand;

    fn plan(
        planner: &mut Self::Planner,
        measured: &Counters,
        inputs: &Self::Inputs,
        ledger: &mut Ledger,
        current: &Self::Streams,
    ) -> Self::Demand;

    fn active(measured: &Counters, work: &HostWork) -> bool;

    fn claim(order: &mut PassOrder) -> Self::Passes;

    fn build(context: &GpuContext, streams: &impl Resources, passes: Self::Passes)
    -> Self::Runtime;

    fn frame(facts: &StepFacts, inputs: &Self::Inputs) -> Self::Frame;

    fn record(
        runtime: &Self::Runtime,
        phase: Phase,
        schedule: &mut Schedule,
        encoder: &mut CommandEncoder,
        streams: &impl Resources,
        frame: &Self::Frame,
    );
}
