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
use registry::{Activity, Live, Plan, Rest, Resting, StepPasses, Streams};
use segment::{Arrival, SegmentTransport};
use wgpu::SubmissionIndex;

pub(crate) use registry::StepFrames;

use crate::World;

/// The scene facts the immovable half of the grid index is derived from. Every one of them can move
/// the grid resolution, and an index derived at another resolution must be derived again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Resolution {
    pub(crate) colliders: u32,
    pub(crate) movable_colliders: u32,
    pub(crate) particles: u32,
}

/// The immovable half of the broadphase grid index: a prefix of the entry streams that the device
/// derives once and reuses until the host declares it stale.
#[derive(Clone, Copy)]
pub(crate) struct Immovable {
    entries: u32,
    rebuild: bool,
    pending: bool,
    resolution: Option<Resolution>,
}

impl Immovable {
    pub(crate) const fn new(entries: u32) -> Self {
        Self {
            entries,
            rebuild: true,
            pending: true,
            resolution: None,
        }
    }

    pub(crate) const fn entries(&self) -> u32 {
        self.entries
    }

    pub(crate) const fn rebuild(&self) -> bool {
        self.rebuild
    }

    pub(crate) fn install(&mut self, entries: u32) {
        self.entries = entries;
        self.invalidate();
        self.pending = true;
    }

    pub(crate) fn invalidate(&mut self) {
        self.rebuild = true;
    }

    pub(crate) fn reconcile(&mut self, resolution: Resolution) {
        if self.resolution != Some(resolution) {
            self.resolution = Some(resolution);
            self.invalidate();
        }
    }

    pub(crate) fn advance(&mut self) {
        self.rebuild = self.pending;
        self.pending = false;
    }
}

pub(crate) struct Backend {
    pub(crate) gpu: GpuContext,
    pub(crate) streams: Streams,
    pub(crate) readback: ReadbackBuffers,
    pub(crate) segments: SegmentTransport,
    pub(crate) passes: StepPasses,
    pub(crate) settling: Settling,
    pub(crate) immovable: Immovable,
    pub(crate) resting: Resting,
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
        let immovable = Immovable::new(plan.broadphase.immovable);
        write_entry_base(&streams, &gpu, immovable.entries());
        seed_resting_counters(&streams, &gpu);
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
            immovable,
            resting: Resting::IDLE,
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

fn seed_resting_counters(streams: &Streams, gpu: &GpuContext) {
    let counters = streams.state.counters.gpu();
    for counter in [
        dynamis_abi::COUNTER_RESTING_ENTRIES,
        dynamis_abi::COUNTER_RESTING_LEVELS,
    ] {
        counters.write_at(
            gpu.queue(),
            counter as u64 * dynamis_abi::COUNTER_STRIDE,
            &0u32.to_le_bytes(),
        );
    }
}

fn write_entry_base(streams: &Streams, gpu: &GpuContext, base: u32) {
    streams
        .state
        .entry_base
        .write(gpu.queue(), bytemuck::cast_slice(&[base]));
}

impl World {
    pub fn submissions(&self) -> u64 {
        self.backend.submissions
    }

    pub(crate) fn submit(&mut self, encoder: SubmissionEncoder) -> SubmissionIndex {
        self.backend.submissions += 1;
        encoder.submit(self.backend.gpu.queue())
    }

    pub(crate) fn invalidate_immovable(&mut self) {
        self.backend.immovable.invalidate();
    }

    pub(crate) fn apply_plan(&mut self, live: &Live) {
        let activity = Activity::of(&self.backend.measured, live, &self.host_work());
        let release = self
            .backend
            .settling
            .release(self.clock.step, activity.busy());
        let plan = Plan::of(
            &self.backend.measured,
            live,
            &self.backend.streams,
            self.backend.immovable.entries(),
            release,
        );
        let immovable = plan.broadphase.immovable;
        if self.backend.immovable.entries() != immovable {
            self.backend.immovable.install(immovable);
            write_entry_base(&self.backend.streams, &self.backend.gpu, immovable);
        }
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
        self.constraints.schedule.publish();
        self.backend
            .readback
            .reserve(&device, &self.backend.streams);
        self.backend
            .segments
            .reserve(&device, &self.backend.streams);
        self.submit(encoder);
        self.backend.resting.invalidate();
    }

    pub(crate) fn reconcile_layout(&mut self, live: &mut Live) {
        let resolution = Resolution {
            colliders: live.broadphase.colliders,
            movable_colliders: live.broadphase.movable_colliders,
            particles: live.broadphase.particles,
        };
        self.backend.immovable.reconcile(resolution);
        let rebuild = self.backend.immovable.rebuild();
        let base = self.backend.immovable.entries();
        let entries = self.backend.streams.broadphase.entry_keys.slots();
        assert!(
            entries >= base,
            "the grid entry stream holds the {base} immovable entries it reserves",
        );
        live.broadphase.entry_base = base;
        live.broadphase.moving_slots = entries - base;
        live.broadphase.immovable_rebuild = rebuild;
        live.rigid.immovable_rebuild = rebuild;
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
