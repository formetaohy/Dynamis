mod resource;
mod schedule;
mod stage;
mod streams;

pub(crate) use resource::{ResourceId, Resources, SlotRef};
pub(crate) use schedule::{Schedule, domain_passes};
pub(crate) use stage::{
    Dispatch, MAX_DISPATCH_WORKGROUPS, Program, Stage, WORKGROUP_SIZE, entry_rows, entry_stream,
    workgroups_of,
};
pub(crate) use streams::{stream_usage, streams};

use wgpu::BufferUsages;

pub(crate) const STREAM: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);
pub(crate) const UNIFORM: BufferUsages = BufferUsages::UNIFORM.union(BufferUsages::COPY_DST);
pub(crate) const PACK: BufferUsages = BufferUsages::COPY_DST.union(BufferUsages::COPY_SRC);

#[cfg(feature = "profile")]
use dynamis_gpu::SubmissionEncoder;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use wgpu::CommandEncoder;

pub(crate) struct Engine {
    per_row: u32,
    schedule: Schedule,
    #[cfg(feature = "profile")]
    timer: Option<dynamis_gpu::GpuTimer>,
}

impl Engine {
    pub(crate) fn new(
        context: &GpuContext,
        schedule: Schedule,
        #[cfg(feature = "profile")] label: &str,
    ) -> Self {
        assert!(
            !schedule.labels().is_empty(),
            "a step schedule needs at least one pass label"
        );
        #[cfg(feature = "profile")]
        let labels = schedule.labels().to_vec();
        Self {
            per_row: context.workgroups_per_row(),
            schedule,
            #[cfg(feature = "profile")]
            timer: context.supports_pass_timing().then(|| {
                dynamis_gpu::GpuTimer::new(
                    context.device(),
                    &labels,
                    context.timestamp_period_ns(),
                    label,
                )
            }),
        }
    }

    pub(crate) const fn per_row(&self) -> u32 {
        self.per_row
    }

    pub(crate) fn open<'a>(
        &'a self,
        encoder: &'a mut CommandEncoder,
        pass: usize,
    ) -> ComputeRecorder<'a> {
        let label = self
            .schedule
            .labels()
            .get(pass)
            .copied()
            .unwrap_or_else(|| panic!("pass {pass} is not part of the step schedule"));
        #[cfg(feature = "profile")]
        if let Some(timer) = &self.timer {
            return ComputeRecorder::begin_timed(
                encoder,
                label,
                Some(timer.writes(pass)),
                self.per_row,
            );
        }
        ComputeRecorder::begin(encoder, label, self.per_row)
    }

    #[cfg(feature = "profile")]
    pub(crate) fn capture_timings(
        &mut self,
        encoder: &mut SubmissionEncoder,
        sequence: u64,
    ) -> Option<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        self.timer
            .as_mut()
            .and_then(|timer| timer.capture(encoder, sequence))
    }

    #[cfg(feature = "profile")]
    pub(crate) fn collect_timings(&mut self) -> Vec<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        match &mut self.timer {
            Some(timer) => timer.collect(),
            None => Vec::new(),
        }
    }
}
