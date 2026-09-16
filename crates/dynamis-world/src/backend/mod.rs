pub(crate) mod archive;
mod readback;
pub(crate) mod registry;
pub(crate) mod segment;

pub use registry::StreamCapacity;

use dynamis_domain::Settling;
#[cfg(feature = "profile")]
use dynamis_gpu::GpuPassTiming;
use dynamis_gpu::{GpuContext, SubmissionEncoder};
use readback::ReadbackBuffers;
use registry::{Activity, Live, Plan, Rest, StepPasses, Streams};
use segment::{Arrival, SegmentTransport};
use wgpu::SubmissionIndex;

pub(crate) use registry::StepFrames;

use crate::World;

pub(crate) struct Backend {
    pub(crate) gpu: GpuContext,
    pub(crate) streams: Streams,
    pub(crate) readback: ReadbackBuffers,
    pub(crate) segments: SegmentTransport,
    pub(crate) passes: StepPasses,
    pub(crate) settling: Settling,
    pub(crate) measured: dynamis_abi::Counters,
    pub(crate) measured_step: Option<u64>,
    pub(crate) written_params: Option<dynamis_abi::StepParamsRecord>,
    pub(crate) published: bool,
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
        let readback = ReadbackBuffers::new(gpu.device(), &streams);
        let segments = SegmentTransport::new(gpu.device(), &streams);
        let passes = StepPasses::new(&gpu, &streams);
        Self {
            gpu,
            streams,
            readback,
            segments,
            passes,
            settling: Settling::IDLE,
            measured: [0; dynamis_abi::COUNTER_COUNT],
            measured_step: None,
            written_params: None,
            published: false,
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
        let activity = Activity::of(&self.backend.measured, live, &self.host_work());
        let release = self
            .backend
            .settling
            .release(self.clock.step, activity.busy());
        let plan = Plan::of(&self.backend.measured, live, &self.backend.streams, release);
        if self.backend.streams.matches(&plan) {
            return;
        }
        let arrivals = self.retire_pending_segments();
        self.consume_segments(arrivals);
        let device = self.backend.gpu.device().clone();
        let mut encoder = SubmissionEncoder::new(&device, "dynamis buffer plan");
        let streams = self.backend.streams.reserve(&device, &mut encoder, &plan);
        assert!(
            streams,
            "a buffer plan that changes capacity must reallocate"
        );
        self.backend
            .readback
            .reserve(&device, &self.backend.streams);
        self.backend
            .segments
            .reserve(&device, &self.backend.streams);
        self.submit(encoder);
    }

    pub(crate) fn copy_segments(&mut self, encoder: &mut SubmissionEncoder) -> Vec<Arrival> {
        let now = self.clock.step;
        self.backend
            .segments
            .copy(encoder, &self.backend.streams, now)
    }

    fn retire_pending_segments(&mut self) -> Vec<Arrival> {
        let mut arrivals = Vec::new();
        if self.backend.segments.pending() {
            let device = self.backend.gpu.device().clone();
            let mut encoder = SubmissionEncoder::new(&device, "dynamis segment readback");
            arrivals.extend(self.copy_segments(&mut encoder));
            self.submit(encoder);
        }
        arrivals
    }

    pub(crate) fn retire_device_facts(&mut self) {
        let mut arrivals = self.retire_pending_segments();
        arrivals.extend(self.backend.segments.drain());
        self.consume_segments(arrivals);
        self.retire_fact_buffers();
    }
}
