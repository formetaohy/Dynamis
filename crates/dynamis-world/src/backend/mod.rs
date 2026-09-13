pub(crate) mod archive;
mod readback;
pub(crate) mod registry;

pub use registry::StreamCapacity;

#[cfg(feature = "profile")]
use dynamis_gpu::GpuPassTiming;
use dynamis_gpu::{GpuContext, SubmissionEncoder};
use readback::ReadbackBuffers;
use registry::{Live, Plan, Planning, Rest, StepPasses, Streams};
use std::collections::VecDeque;
use wgpu::SubmissionIndex;

pub(crate) use registry::StepFrames;

use crate::World;

pub(crate) struct Backend {
    pub(crate) gpu: GpuContext,
    pub(crate) streams: Streams,
    pub(crate) readback: ReadbackBuffers,
    pub(crate) passes: StepPasses,
    pub(crate) planning: Planning,
    pub(crate) measured: dynamis_abi::Counters,
    pub(crate) declared: VecDeque<(u64, dynamis_abi::DeclaredCounters)>,
    pub(crate) measured_step: Option<u64>,
    pub(crate) working: bool,
    pub(crate) rest: Rest,
    pub(crate) inspect: Option<dynamis_gpu::Readback>,
    pub(crate) submissions: u64,
    #[cfg(feature = "profile")]
    pub(crate) pass_timings: Vec<GpuPassTiming>,
}

impl Backend {
    pub(crate) fn new(gpu: GpuContext) -> Self {
        let plan = Plan::minimum();
        let streams = Streams::new(gpu.device(), gpu.queue(), &plan);
        let readback = ReadbackBuffers::new(gpu.device(), &plan);
        let passes = StepPasses::new(&gpu, &streams);
        Self {
            gpu,
            streams,
            readback,
            passes,
            planning: Planning::new(),
            measured: [0; dynamis_abi::COUNTER_COUNT],
            declared: VecDeque::new(),
            measured_step: None,
            working: false,
            rest: Rest::IDLE,
            inspect: None,
            submissions: 0,
            #[cfg(feature = "profile")]
            pass_timings: Vec::new(),
        }
    }
}

impl World {
    pub fn submissions(&self) -> u64 {
        self.backend.submissions
    }

    pub(crate) fn submit(&mut self, encoder: SubmissionEncoder) -> SubmissionIndex {
        self.backend.submissions += 1;
        encoder.submit(self.backend.gpu.queue())
    }

    pub(crate) fn apply_plan(&mut self, live: &Live) {
        let plan = self
            .backend
            .planning
            .plan(&self.backend.measured, live, &self.backend.streams);
        if self.backend.streams.matches(&plan) && self.backend.readback.matches(&plan) {
            return;
        }
        if !self.backend.readback.matches(&plan) {
            self.sync_events();
            self.drain_readbacks();
        }
        let device = self.backend.gpu.device().clone();
        let mut encoder = SubmissionEncoder::new(&device, "dynamis buffer plan");
        let streams = self.backend.streams.reserve(&device, &mut encoder, &plan);
        let readback = self.backend.readback.reserve(&device, &plan);
        assert!(
            streams | readback,
            "a buffer plan that changes capacity must reallocate"
        );
        self.submit(encoder);
    }
}
