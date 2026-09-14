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
    timed: bool,
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
            timed: true,
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

    pub fn ran(&self, index: u32) -> bool {
        self.opened.get(index as usize).copied().unwrap_or(false)
    }

    pub fn ran_labels(&self) -> Vec<&'static str> {
        self.pipeline
            .passes()
            .iter()
            .enumerate()
            .filter(|(index, _)| self.ran(*index as u32))
            .map(|(_, pass)| pass.label)
            .collect()
    }

    pub fn begin_step(&mut self) {
        self.begin(true);
    }

    pub fn begin_query(&mut self) {
        self.begin(false);
    }

    fn begin(&mut self, timed: bool) {
        self.opened.fill(false);
        self.last = None;
        self.timed = timed;
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
        if self.timed
            && let Some(timer) = &self.timer
        {
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
