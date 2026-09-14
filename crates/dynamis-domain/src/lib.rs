mod capacity;
mod facts;
mod registry;
mod streams;

pub use capacity::{MIN_SLOTS, STREAM_FLOOR, StreamWatch, grown, product, settled};
pub use facts::StepFacts;
pub use streams::DomainStreams;

use dynamis_abi::Counters;
use dynamis_gpu::GpuContext;
use dynamis_gpu::Resources;
use dynamis_pass::{PassGroup, Pipeline, Schedule};
use wgpu::CommandEncoder;

pub trait Domain {
    const ID: u32;

    const SIMULATES: bool;

    const PASS_EDGES: dynamis_pass::PassEdges;

    type Demand: Copy;
    type Inputs: Copy;
    type Work: Copy;
    type Streams: DomainStreams<Demand = Self::Demand>;
    type Planner: Default;
    type Passes;
    type Runtime;
    type Frame: Copy;
    type Capacity: Copy;

    fn minimum() -> Self::Demand;

    fn occupied(inputs: &Self::Inputs) -> bool;

    fn pending(work: &Self::Work) -> bool;

    fn active(measured: &Counters) -> bool;

    fn pass_groups() -> &'static [PassGroup];

    fn resolve(pipeline: &Pipeline) -> Self::Passes;

    fn build(context: &GpuContext, streams: &impl Resources, passes: Self::Passes)
    -> Self::Runtime;

    fn gates(frame: &Self::Frame) -> u8;

    fn frame(facts: &StepFacts, inputs: &Self::Inputs) -> Self::Frame;

    fn capacity(streams: &Self::Streams) -> Self::Capacity;

    fn record(
        runtime: &Self::Runtime,
        pass: u32,
        schedule: &mut Schedule,
        encoder: &mut CommandEncoder,
        streams: &impl Resources,
        frame: &Self::Frame,
    );
}
