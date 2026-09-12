use super::World;
use crate::dynamics::Pipeline;
use crate::dynamics::streams::{Planning, Streams};
#[cfg(feature = "profile")]
use dynamis_gpu::GpuPassTiming;
use dynamis_gpu::{GpuContext, SubmissionEncoder};
use wgpu::SubmissionIndex;

pub(crate) struct Backend {
    pub(crate) gpu: GpuContext,
    pub(crate) streams: Streams,
    pub(crate) pipeline: Pipeline,
    pub(crate) planning: Planning,
    pub(crate) measured: dynamis_layout::Counters,
    pub(crate) measured_step: Option<u64>,
    pub(crate) commanded_step: Option<u64>,
    pub(crate) inspect: Option<dynamis_gpu::Readback>,
    pub(crate) submissions: u64,
    #[cfg(feature = "profile")]
    pub(crate) pass_timings: Vec<GpuPassTiming>,
}

impl Backend {
    pub(crate) fn new(gpu: GpuContext) -> Self {
        let plan = Planning::minimum();
        let streams = Streams::new(gpu.device(), gpu.queue(), &plan);
        let pipeline = Pipeline::new(&gpu, &streams);
        Self {
            gpu,
            streams,
            pipeline,
            planning: Planning::new(),
            measured: [0; dynamis_layout::COUNTER_COUNT],
            measured_step: None,
            commanded_step: None,
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

    pub(crate) fn apply_plan(&mut self) {
        let live = self.live();
        let plan = self
            .backend
            .planning
            .plan(&self.backend.measured, &live, &self.backend.streams);
        if self.backend.streams.matches(&plan) {
            return;
        }
        if !self.backend.streams.readback_matches(&plan) {
            self.sync_events();
            self.drain_readbacks();
        }
        let device = self.backend.gpu.device().clone();
        let mut encoder = SubmissionEncoder::new(&device, "dynamis buffer plan");
        assert!(
            self.backend.streams.reserve(&device, &mut encoder, &plan),
            "a buffer plan that changes capacity must reallocate"
        );
        self.submit(encoder);
    }
}
