use crate::{Pass, Pipeline};
#[cfg(feature = "profile")]
use dynamis_gpu::SubmissionEncoder;
use dynamis_gpu::{ComputeRecorder, GpuContext};
use wgpu::CommandEncoder;

pub struct Schedule {
    pipeline: Pipeline,
    per_row: u32,
    opened: Vec<bool>,
    last: Option<u32>,
    #[cfg(feature = "profile")]
    timer: Option<dynamis_gpu::GpuTimer>,
}

impl Schedule {
    pub fn new(
        context: &GpuContext,
        pipeline: Pipeline,
        #[cfg(feature = "profile")] label: &str,
    ) -> Self {
        assert!(
            !pipeline.is_empty(),
            "a step schedule needs at least one pass"
        );
        #[cfg(feature = "profile")]
        let labels = pipeline
            .passes()
            .iter()
            .map(|pass| pass.label)
            .collect::<Vec<_>>();
        Self {
            per_row: context.workgroups_per_row(),
            opened: vec![false; pipeline.len()],
            last: None,
            pipeline,
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

    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    pub fn pass(&self, index: u32) -> Pass {
        self.pipeline.pass(index)
    }

    pub fn begin_step(&mut self) {
        self.opened.fill(false);
        self.last = None;
    }

    pub fn open<'a>(
        &'a mut self,
        encoder: &'a mut CommandEncoder,
        pass: u32,
    ) -> ComputeRecorder<'a> {
        let declared = self.pipeline.pass(pass);
        assert!(
            !self.opened[pass as usize],
            "pass {:?} opens twice in one step",
            declared.label
        );
        if let Some(last) = self.last {
            assert!(
                last < pass,
                "pass {:?} opens after {:?} in the resolved step",
                declared.label,
                self.pipeline.pass(last).label
            );
        }
        self.opened[pass as usize] = true;
        self.last = Some(pass);
        #[cfg(feature = "profile")]
        if let Some(timer) = &self.timer {
            return ComputeRecorder::begin_timed(
                encoder,
                declared.label,
                Some(timer.writes(pass as usize)),
                self.per_row,
            );
        }
        ComputeRecorder::begin(encoder, declared.label, self.per_row)
    }

    #[cfg(feature = "profile")]
    pub fn capture_timings(
        &mut self,
        encoder: &mut SubmissionEncoder,
    ) -> Option<Vec<dynamis_gpu::GpuPassTiming>> {
        let ran = self.opened.clone();
        self.timer
            .as_mut()
            .and_then(|timer| timer.capture(encoder, &ran))
    }

    #[cfg(feature = "profile")]
    pub fn collect_timings(&mut self) -> Vec<Vec<dynamis_gpu::GpuPassTiming>> {
        match &mut self.timer {
            Some(timer) => timer.collect(),
            None => Vec::new(),
        }
    }
}
