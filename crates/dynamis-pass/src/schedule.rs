use crate::{Pass, PassOrder, Phase};
#[cfg(feature = "profile")]
use dynamis_gpu::SubmissionEncoder;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use wgpu::CommandEncoder;

pub struct Schedule {
    order: PassOrder,
    per_row: u32,
    opened: Vec<bool>,
    phase: Option<Phase>,
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
            !order.passes().is_empty(),
            "a step schedule needs at least one pass"
        );
        #[cfg(feature = "profile")]
        let labels = order
            .passes()
            .iter()
            .map(|pass| pass.label)
            .collect::<Vec<_>>();
        Self {
            per_row: context.workgroups_per_row(),
            opened: vec![false; order.passes().len()],
            phase: None,
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

    pub fn declared(&self) -> &[Pass] {
        self.order.passes()
    }

    pub fn begin_step(&mut self) {
        self.opened.fill(false);
        self.phase = None;
    }

    pub fn open<'a>(
        &'a mut self,
        encoder: &'a mut CommandEncoder,
        pass: usize,
    ) -> ComputeRecorder<'a> {
        let declared = *self
            .order
            .passes()
            .get(pass)
            .unwrap_or_else(|| panic!("pass {pass} is not part of the step schedule"));
        assert!(
            !self.opened[pass],
            "pass {:?} opens twice in one step",
            declared.label
        );
        assert!(
            self.phase.is_none_or(|phase| phase <= declared.phase),
            "pass {:?} opens in {:?} after {:?}",
            declared.label,
            declared.phase,
            self.phase.expect("an open pass declares a phase")
        );
        self.opened[pass] = true;
        self.phase = Some(declared.phase);
        #[cfg(feature = "profile")]
        if let Some(timer) = &self.timer {
            return ComputeRecorder::begin_timed(
                encoder,
                declared.label,
                Some(timer.writes(pass)),
                self.per_row,
            );
        }
        ComputeRecorder::begin(encoder, declared.label, self.per_row)
    }

    #[cfg(feature = "profile")]
    pub fn capture_timings(
        &mut self,
        encoder: &mut SubmissionEncoder,
        sequence: u64,
    ) -> Option<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        let ran = self.opened.clone();
        self.timer
            .as_mut()
            .and_then(|timer| timer.capture(encoder, sequence, &ran))
    }

    #[cfg(feature = "profile")]
    pub fn collect_timings(&mut self) -> Vec<(u64, Vec<dynamis_gpu::GpuPassTiming>)> {
        match &mut self.timer {
            Some(timer) => timer.collect(),
            None => Vec::new(),
        }
    }
}
