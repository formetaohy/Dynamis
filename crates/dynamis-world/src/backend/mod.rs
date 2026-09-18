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
use crate::derivation::{GridStorage, ImmovableGrid, JointOrderStorage, RestingGrid};

/// The immovable half of the broadphase grid index: a prefix of the entry streams that the device
/// holds until the facts its entries are built from, the range it reserved, or the storage it lives
/// in has moved. The index is a pure function of those facts, so a step that runs neither the
/// derivation nor the emission simply leaves the difference standing until a step that does.
#[derive(Clone, Copy)]
pub(crate) struct Immovable {
    entries: u32,
    held: Option<ImmovableGrid>,
    due: Option<ImmovableGrid>,
}

impl Immovable {
    pub(crate) const fn new(entries: u32) -> Self {
        Self {
            entries,
            held: None,
            due: None,
        }
    }

    pub(crate) const fn entries(&self) -> u32 {
        self.entries
    }

    pub(crate) fn rebuild(&self) -> bool {
        self.held.is_none() || self.due != self.held
    }

    pub(crate) fn install(&mut self, entries: u32) {
        self.entries = entries;
    }

    /// Records the facts at hand and answers whether the device still owes the entries they answer.
    pub(crate) fn reconcile(&mut self, grid: ImmovableGrid) -> bool {
        self.due = Some(grid);
        self.rebuild()
    }

    /// Records that the step ran the derivation, so the device now holds the entries the facts at
    /// hand answer.
    pub(crate) fn derived(&mut self) {
        self.held = self.due;
    }
}

/// The resting half of the broadphase grid index: the entries of every sleeping body. Resting
/// entries are keyed at a resolution the device derives, so they are derived again whenever that
/// resolution moves, whenever the facts behind them move, and once their region grows old enough
/// that the device counters can no longer vouch for it.
#[derive(Clone, Copy)]
pub(crate) struct Resting {
    held: Option<RestingGrid>,
    due: Option<RestingGrid>,
    derived: u32,
}

impl Resting {
    pub(crate) const IDLE: Self = Self {
        held: None,
        due: None,
        derived: 0,
    };

    pub(crate) const REGION_LIFETIME: u32 = 64;

    /// Records the facts at hand and answers whether the device still owes the entries they answer.
    pub(crate) fn reconcile(&mut self, step: u64, grid: RestingGrid) -> bool {
        self.due = Some(grid);
        self.held.is_none()
            || self.held != self.due
            || step.wrapping_sub(u64::from(self.derived)) >= u64::from(Self::REGION_LIFETIME)
    }

    /// Records that the step ran the derivation, so the device now holds the entries the facts at
    /// hand answer and the region's lifetime starts over.
    pub(crate) fn derived(&mut self, step: u64) {
        self.held = self.due;
        self.derived = step as u32;
    }
}

/// The derivations a step owes the device. A run that indexes the scene derives them, so what it
/// owed is what the device holds from then on.
#[derive(Clone, Copy)]
pub(crate) struct Derived {
    pub(crate) immovable: bool,
    pub(crate) resting: bool,
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
        self.backend
            .readback
            .reserve(&device, &self.backend.streams);
        self.backend
            .segments
            .reserve(&device, &self.backend.streams);
        self.submit(encoder);
    }

    /// Derives every device index the host maintains from the facts at hand, and answers whether
    /// the resting region is owed a rebuild this step. Each index compares the inputs it recorded
    /// against the facts the step presents, so a mutation, a capacity plan, or a snapshot install
    /// that replaces the storage behind an index is seen without any of them having to announce it.
    pub(crate) fn reconcile_derivations(&mut self, live: &mut Live) -> Derived {
        let facts = self.facts;
        let measured = self.resting_resolution();
        let storage = GridStorage {
            keys: self.backend.streams.broadphase.entry_keys.identity(),
            order: self.backend.streams.broadphase.entry_order.identity(),
            entries: self.backend.streams.broadphase.entries.identity(),
        };
        let resolution = crate::derivation::Resolution {
            colliders: live.broadphase.colliders,
            movable_colliders: live.broadphase.movable_colliders,
            particles: live.broadphase.particles,
        };
        let reservation = self.backend.immovable.entries();
        let immovable = self.backend.immovable.reconcile(ImmovableGrid {
            colliders: facts.colliders,
            shapes: facts.shapes,
            immovable_edits: facts.immovable_edits,
            resolution,
            reservation,
            storage,
        });
        let resting = self.backend.resting.reconcile(
            self.clock.step,
            RestingGrid {
                colliders: facts.colliders,
                shapes: facts.shapes,
                poses: facts.poses,
                layout: facts.layout,
                activity: facts.activity,
                resolution: measured,
                storage,
            },
        );
        let entries = self.backend.streams.broadphase.entry_keys.slots();
        assert!(
            entries >= reservation,
            "the grid entry stream holds the {reservation} immovable entries it reserves",
        );
        live.broadphase.entry_base = reservation;
        live.broadphase.moving_slots = entries - reservation;
        live.broadphase.immovable_rebuild = immovable;
        live.broadphase.resting_rebuild = resting;
        live.rigid.immovable_rebuild = immovable;
        live.rigid.resting_rebuild = resting;
        Derived { immovable, resting }
    }

    pub(crate) fn joint_order_storage(&self) -> JointOrderStorage {
        let streams = &self.backend.streams.rigid;
        JointOrderStorage {
            rows: streams.joint_rows.identity(),
            layers: streams.joint_layers.identity(),
            components: streams.joint_components.identity(),
            batches: streams.joint_batches.identity(),
        }
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
