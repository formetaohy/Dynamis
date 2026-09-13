use crate::PassOrder;
#[cfg(feature = "profile")]
use dynamis_gpu::SubmissionEncoder;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use wgpu::CommandEncoder;

pub struct Schedule {
    order: PassOrder,
    per_row: u32,
    #[cfg(feature = "profile")]
    timer: Option<dynamis_gpu::GpuTimer>,
}

impl Schedule {
    pub fn new(
        context: &GpuContext,
        order: PassOrder,
        #[cfg(feature = "profile")] label: &str,
    ) -> Self {
        assert!(
            !order.labels().is_empty(),
            "a step schedule needs at least one pass label"
        );
        #[cfg(feature = "profile")]
        let labels = order.labels().to_vec();
        Self {
            per_row: context.workgroups_per_row(),
            order,
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

    pub const fn per_row(&self) -> u32 {
        self.per_row
    }

    pub fn open<'a>(&'a self, encoder: &'a mut CommandEncoder, pass: usize) -> ComputeRecorder<'a> {
        let label = self
            .order
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
    pub fn capture_timings(
        &mut self,
        encoder: &mut SubmissionEncoder,
        sequence: u64,
    ) -> Option<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        self.timer
            .as_mut()
            .and_then(|timer| timer.capture(encoder, sequence))
    }

    #[cfg(feature = "profile")]
    pub fn collect_timings(&mut self) -> Vec<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        match &mut self.timer {
            Some(timer) => timer.collect(),
            None => Vec::new(),
        }
    }
}
