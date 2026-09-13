mod facts;
mod registry;

pub use facts::{Run, StepFacts};

use dynamis_abi::Counters;
use dynamis_gpu::GpuContext;
use dynamis_pass::{PassOrder, Phase, Resources, Schedule};
use wgpu::CommandEncoder;

pub trait Domain {
    const ID: u32;

    type Demand: Copy;
    type Inputs: Copy;
    type Work: Copy;
    type Streams;
    type Planner: Default;
    type Passes;
    type Runtime;
    type Frame: Copy;
    type Capacity: Copy;

    fn minimum() -> Self::Demand;

    fn active(measured: &Counters, work: &Self::Work) -> bool;

    fn claim(order: &mut PassOrder) -> Self::Passes;

    fn build(context: &GpuContext, streams: &impl Resources, passes: Self::Passes)
    -> Self::Runtime;

    fn frame(facts: &StepFacts, inputs: &Self::Inputs, run: Run) -> Self::Frame;

    fn capacity(streams: &Self::Streams) -> Self::Capacity;

    fn record(
        runtime: &Self::Runtime,
        phase: Phase,
        schedule: &mut Schedule,
        encoder: &mut CommandEncoder,
        streams: &impl Resources,
        frame: &Self::Frame,
    );
}
